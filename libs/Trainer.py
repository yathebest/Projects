import os.path
from copy import deepcopy
from math import ceil
from typing import Optional, Literal

import polars as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm, trange

from libs.models.BaseRecurrentModel import BaseRecurrentModel
from .constants import *
from .utils import count_polars, build_mask, build_embeddings_map, build_sequences_from_map


class Trainer:
    """
    :param model: Recurrent model to predict embeddings
    :param train_interactions: LazyFrame containing train interactions with columns [USER, ITEM, TARGET, TIME_INDEX]
    :param val_interactions: LazyFrame containing val interactions with columns [USER, ITEM, TARGET, TIME_INDEX]
    :param items_metadata: LazyFrame containing all items metadata with columns [ITEM, EMBEDDING]
    :param optimizer: optional torch optimizer. Will be created default if not provided and create_optimizer is not called
    :param loss_type: prefix 'pair_' = pairwise (no margin), 'triplet_' = margin-based (uses self.loss_margin and negatives)
    :param loss_margin: used for margin-based losses (with negative samples)
    :param negative_loss_pooling: used to pool negative loss from K to 1 ('max' | 'mean' | 'min')
    :param negative_ratio: used to sample negatives. For 'cross-entropy' negative_ratio must be > 0
    :param num_recent_videos: number of known the most recent interactions to keep for each user
    """
    def __init__(self, model: BaseRecurrentModel,
                 train_interactions: pl.LazyFrame, val_interactions: pl.LazyFrame,
                 items_metadata: pl.LazyFrame,
                 optimizer: Optional[Optimizer] = None,
                 num_recent_videos: int = 500,
                 loss_type: Literal['pair_mse', 'triplet_mse', 'pair_cos', 'triplet_cos', 'ce'] = 'pair_mse',
                 loss_margin: float = 0.0,
                 negative_ratio: int = 0,
                 negative_loss_pooling: Literal['max', 'mean', 'min'] = 'mean',
                 device: torch.device | str = 'cpu',
                 ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.start_epoch = 1
        self.items_df = items_metadata
        self.loss_type = loss_type
        self.loss_margin = loss_margin
        self.negative_ratio = negative_ratio
        self.negative_loss_pooling = negative_loss_pooling
        self.device = device
        self.history = None

        def _agg(interactions: pl.LazyFrame):
            return interactions \
                .sort(by=[pl.col(USER), pl.col(TIME_INDEX)], descending=[False, True]) \
                .group_by(USER) \
                .agg([pl.col(ITEM).slice(0, num_recent_videos).alias(ITEM),
                      pl.col(TIME_INDEX).slice(0, num_recent_videos).alias(TIME_INDEX),
                      pl.col(TARGET).slice(0, num_recent_videos).alias(TARGET)])

        self.data = _agg(pl.concat([train_interactions, val_interactions]))
        self.train = _agg(train_interactions)
        self.val = _agg(val_interactions)

        print("Count")
        self.data_count = count_polars(self.data)
        self.train_count = count_polars(self.train)
        self.val_count = count_polars(self.val)

        print(f"Users: All: {self.data_count:_} Train: {self.train_count:_} Val: {self.val_count:_}")

    def create_optimizer(self, cls: str = 'AdamW', *args, **kwargs):
        self.optimizer = getattr(torch.optim, cls)(self.model.parameters(), *args, **kwargs)

    def load_checkpoint(self, path: str = "./checkpoints/model.pt"):
        chkpt = torch.load(path)
        self.model.load_state_dict(chkpt["model_dict"])
        self.optimizer.load_state_dict(chkpt["optimizer_dict"])
        self.start_epoch = chkpt["epoch"] + 1
        self.history = chkpt["history"]

    def train_epoch(self, users_batch_size: int = 4096, epoch: Optional[int] = None, verbose=True) -> float:
        self.model.train()

        total_batches = max(1, ceil(self.train_count / users_batch_size)) if self.train_count else 1
        running_loss = 0.0
        processed_batches = 0

        bar: tqdm_asyncio[pl.DataFrame] = tqdm(
            self.train.collect_batches(chunk_size=users_batch_size),
            total=total_batches,
            disable=not verbose,
            desc=f"Train Epoch: {epoch}" if epoch is not None else "Train Epoch"
        )

        for data_batch in bar:
            predict = self.model.process_data_batch(data_batch, items_df=self.items_df, mode='train')
            predict = predict[:-1]

            target, mask, neg = self._build_target(data_batch[ITEM].to_list())
            target, mask = target[1:], mask[1:]
            if neg is not None:
                neg = neg[1:]

            batch_loss = self._compute_loss(predict, target, mask, neg)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            running_loss += float(batch_loss.item())
            processed_batches += 1

            avg_loss = running_loss / processed_batches if processed_batches else 0.0
            bar.set_description(f"Train Epoch: {epoch} Loss: {avg_loss:.4f}" if epoch is not None else f"Train Loss: {avg_loss:.4f}")

        return (running_loss / processed_batches) if processed_batches else 0.0

    def val_epoch(self, users_batch_size: int = 4096, epoch: Optional[int] = None, verbose=True) -> float | None:
        if self.val is None:
            return None

        self.model.eval()

        total_batches = max(1, ceil(self.val_count / users_batch_size)) if self.val_count else 1
        running_loss = 0.0
        processed_batches = 0

        bar: tqdm_asyncio[pl.DataFrame] = tqdm(
            self.val.collect_batches(chunk_size=users_batch_size),
            total=total_batches,
            disable=not verbose,
            desc=f"Val Epoch: {epoch}" if epoch is not None else "Val Epoch"
        )

        with torch.no_grad():
            for data_batch in bar:
                predict = self.model.process_data_batch(data_batch, items_df=self.items_df, mode='val')
                predict = predict[:-1]  # t-1

                target, mask, neg = self._build_target(data_batch[ITEM].to_list())
                target, mask = target[1:], mask[1:] # t+1
                if neg is not None:
                    neg = neg[1:]

                batch_loss = self._compute_loss(predict, target, mask, neg)

                running_loss += float(batch_loss.cpu().item())
                processed_batches += 1

                avg_loss = running_loss / processed_batches if processed_batches else 0.0
                bar.set_description(f"Val Epoch: {epoch} Loss: {avg_loss:.4f}" if epoch is not None else f"Val Loss: {avg_loss:.4f}")

        return (running_loss / processed_batches) if processed_batches else 0.0

    def fit(self, epochs: int = 5, users_batch_size: int = 4096, patience: Optional[int] = None,
            save: bool = True, verbose: bool = True,
            save_name = "model.pt"):
        if not self.model.trainable:
            print("Model is not trainable")
            return

        if self.optimizer is None:
            self.create_optimizer()
            print("Default optimizer created")

        self.history: dict[str, list] = {'train_loss': [], 'val_loss': []}
        best = (-1, float('inf'), deepcopy(self.model.state_dict()))
        no_improvement = 0

        bar = trange(self.start_epoch, epochs + 1, disable=verbose)
        try:
            for epoch in bar:
                train_loss = self.train_epoch(users_batch_size=users_batch_size, epoch=epoch, verbose=verbose)
                val_loss = self.val_epoch(users_batch_size=2*users_batch_size, epoch=epoch, verbose=verbose)
                candidate_loss = val_loss or train_loss

                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)

                bar.set_description(f"Epoch: {epoch} Train Loss: {train_loss:.4f}" +
                                    (f" Val Loss: {val_loss:.4f}" if val_loss else "") +
                                    f"Best Epoch: {best[0]}")

                if candidate_loss < best[1]:
                    best = (epoch, candidate_loss, deepcopy(self.model.state_dict()))
                    no_improvement = 0

                    if save:
                        if not os.path.exists("./checkpoints"):
                            os.mkdir("./checkpoints")
                        torch.save({
                            "model_dict": self.model.state_dict(),
                            "optimizer_dict": self.optimizer.state_dict(),
                            "epoch": epoch,
                            "history": self.history,
                        }, f"./checkpoints/{save_name}")

                if no_improvement == patience:
                    print("Early stop")
                    break

            print(f"Completed! Best Loss {best[1]:.4f} at Epoch {best[0]}")
            self.model.load_state_dict(best[2])

        except Exception as e:
            print(f"Exception raised during fit: {e}\nTaking current best model from Epoch {best[0]} with Loss {best[1]:.4f}")
            self.model.load_state_dict(best[2])
            raise

    def _build_target(self, item_lists: list[list]) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        :returns tuple of Tensors: target (T, B, D), mask (T, B), Optional[negatives (T, B, K, D)]
        """
        mapping = build_embeddings_map(self.items_df, set().union(*item_lists))
        target = build_sequences_from_map(mapping, item_lists, batch_first=False, padding_side=self.model.padding_side, device=self.device)
        mask = build_mask(item_lists, batch_first=False, padding_side=self.model.padding_side, device=self.device)
        negatives = self._sample_negatives(mapping, item_lists) if self.negative_ratio > 0 else None

        return target, mask, negatives

    def _sample_negatives(self, embeddings_mapping: dict, item_lists: list[list]):
        """
        :returns: Tensor of shape (T, B, K, D)
        """
        B = len(item_lists)
        T = max((len(s) for s in item_lists), default=0)
        K = self.negative_ratio

        unique_items = list(set().union(*item_lists))
        M = len(unique_items)

        id2idx = {it: idx for idx, it in enumerate(unique_items)}

        emb_list = [torch.from_numpy(embeddings_mapping[it]).to(self.device).float() for it in unique_items]
        emb_matrix = torch.stack(emb_list, dim=0)  # (M, D)

        mask_allowed = torch.ones((B, M), dtype=torch.bool, device=self.device)
        for b, lst in enumerate(item_lists):
            for it in lst:
                idx = id2idx.get(it)
                if idx is not None:
                    mask_allowed[b, idx] = False

        cand_counts = mask_allowed.sum(dim=1)  # (B,)
        valid_users_mask = cand_counts > 0
        valid_users = valid_users_mask.nonzero(as_tuple=False).squeeze(1)

        enough_mask = cand_counts[valid_users] >= K
        users_enough = valid_users[enough_mask]
        users_not_enough = valid_users[~enough_mask]

        def sample_for_users(user_idx_tensor, replacement: bool):
            """
            :returns Tensor (T, len(user_idx_tensor), K) of indicis with sampled local item indices.
            """
            if user_idx_tensor.numel() == 0:
                return None

            weights = mask_allowed[user_idx_tensor].float()
            weights_rep = weights.repeat_interleave(T, dim=0)
            sampled = torch.multinomial(weights_rep, num_samples=K, replacement=replacement)
            sampled = sampled.view(T, user_idx_tensor.numel(), K)
            return sampled

        idxs_enough = sample_for_users(users_enough, replacement=False)
        idxs_not_enough = sample_for_users(users_not_enough, replacement=True)

        idxs_all = torch.full((T, B, K), -1, dtype=torch.long, device=self.device)
        if idxs_enough is not None:
            idxs_all[:, users_enough, :] = idxs_enough
        if idxs_not_enough is not None:
            idxs_all[:, users_not_enough, :] = idxs_not_enough

        idxs_clamped = idxs_all.clamp(min=0)  # (T, B, K)
        neg_emb = emb_matrix[idxs_clamped]  # (T, B, K, D)
        neg_emb[idxs_all < 0] = 0.0

        return neg_emb  # (T, B, K, D)

    def _compute_loss(self, predict: Tensor, target: Tensor, mask: Tensor, negative: Optional[Tensor] = None) -> Tensor:
        """
        :param predict:  (T, B, D)
        :param target:   (T, B, D)
        :param mask:     (T, B) boolean
        :param negative: Optional[(T, B, K, D)]
        """

        def pool_over_k(x: Tensor, dim: int = -1) -> Tensor:
            if self.negative_loss_pooling == 'max':
                return x.max(dim=dim).values
            if self.negative_loss_pooling == 'min':
                return x.min(dim=dim).values
            if self.negative_loss_pooling == 'mean':
                return x.mean(dim=dim)
            raise NotImplementedError()

        T, B, D = predict.shape
        K = negative.shape[2] if negative is not None else None

        predict_flat = predict.reshape(T*B, D)   # (T*B, D)
        target_flat = target.reshape(T*B, D)     # (T*B, D)
        mask_flat = mask.reshape(T*B)               # (T*B)
        N = int(mask_flat.sum().item())

        if '_' in self.loss_type:
            family, base = self.loss_type.split('_', 1)
        else:
            family, base = 'pair', self.loss_type
        if base in ('cross-entropy', 'crossentropy', 'ce'):
            base = 'ce'

        if family == 'pair' and base == 'mse':  # PairwiseMSELoss
            a_pos = predict_flat[mask_flat]                 # (N, D)
            p_pos = target_flat[mask_flat]                  # (N, D)
            sq_pos = (a_pos - p_pos).pow(2).sum(dim=1)      # (N)
            pos_loss = sq_pos.mean()                        # ()

            if negative is not None:
                negative_flat = negative.view(T * B, K, D)              # (T*B, K, D)
                negs_sel = negative_flat[mask_flat]                     # (N, K, D)
                a_exp = a_pos.unsqueeze(1).expand(-1, K, -1)            # (N, K, D)
                neg_dists = torch.norm(a_exp - negs_sel, p=2, dim=2)    # (N, K)
                neg_losses = F.relu(self.loss_margin - neg_dists)       # (N, K)
                pooled_neg = pool_over_k(neg_losses, dim=1)             # (N)
                neg_loss = pooled_neg.mean()                            # ()
                return (pos_loss + neg_loss) / 2                        # ()

            return pos_loss

        if family == 'triplet' and base == 'mse':  # TripletMSELoss
            if negative is None:
                raise NotImplementedError()

            pos_dists = torch.norm(predict - target, p=2, dim=2)                        # (T, B)
            neg_dists = torch.norm(predict.unsqueeze(2) - negative, p=2, dim=3)         # (T, B, K)
            hinge_all = F.relu(self.loss_margin + pos_dists.unsqueeze(2) - neg_dists)   # (T, B, K)
            pooled = pool_over_k(hinge_all, dim=2)                                      # (T, B)
            return pooled[mask].mean()                                                  # ()

        if family == 'pair' and base == 'cos':  # PairwiseCosineEmbeddingLoss
            a_pos = predict_flat[mask_flat]                                 # (N, D)
            p_pos = target_flat[mask_flat]                                  # (N, D)
            labels_pos = torch.ones(a_pos.size(0), device=predict.device)   # (N)
            pos_loss = F.cosine_embedding_loss(                             # ()
                F.normalize(a_pos, p=2, dim=1),
                F.normalize(p_pos, p=2, dim=1),
                labels_pos, margin=0.0, reduction='mean',
            )

            if negative is not None:
                negative_flat = negative.view(T * B, K, D)                          # (T*B, K, D)
                negs_sel = negative_flat[mask_flat]                                 # (N, K, D)
                a_rep = a_pos.unsqueeze(1).expand(-1, K, -1).reshape(-1, D)         # (N*K, D)
                neg_rep = negs_sel.reshape(-1, D)                                   # (N*K, D)
                labels_neg = -torch.ones(a_rep.size(0), device=predict.device)      # (N*K)
                neg_losses_flat = F.cosine_embedding_loss(                          # (N*K)
                    F.normalize(a_rep, p=2, dim=1),
                    F.normalize(neg_rep, p=2, dim=1),
                    labels_neg, margin=0.0, reduction='none'
                )
                neg_losses = neg_losses_flat.view(N, K)         # (N, K)
                pooled_neg = pool_over_k(neg_losses, dim=1)     # (N)
                neg_loss = pooled_neg.mean()                    # ()
                return (pos_loss + neg_loss) / 2                # ()

            return pos_loss

        if family == 'triplet' and base == 'cos':  # TripletCosineEmbeddingLoss
            if negative is None:
                raise NotImplementedError()

            pos_dist = 1.0 - F.cosine_similarity(predict, target, dim=2).unsqueeze(2)   # (T, B, 1)
            pred_n = F.normalize(predict, p=2, dim=2).unsqueeze(2)                      # (T, B, 1, D)
            neg_n = F.normalize(negative, p=2, dim=3)                                   # (T, B, K, D)
            neg_dist_all = 1.0 - (pred_n * neg_n).sum(dim=3)                            # (T, B, K)
            hinge_all = F.relu(self.loss_margin + pos_dist - neg_dist_all)              # (T, B, K)
            pooled = pool_over_k(hinge_all, dim=2)                                      # (T, B)
            return pooled[mask].mean()                                                  # ()

        if base == 'ce':  # CrossEntropyLoss
            if negative is None:
                raise NotImplementedError()

            pos_scores = F.cosine_similarity(predict, target, dim=2).unsqueeze(2)   # (T, B, 1)
            pred_n = F.normalize(predict, p=2, dim=2).unsqueeze(2)                  # (T, B, 1, D)
            neg_n = F.normalize(negative, p=2, dim=3)                               # (T, B, K, D)
            neg_scores = (pred_n * neg_n).sum(dim=3)                                # (T, B, K)
            logits = torch.cat([pos_scores, neg_scores], dim=2)              # (T, B, K+1)

            # positive class 0
            logits_flat = logits.view(T * B, K+1)                               # (T*B, K+1)
            logits_sel = logits_flat[mask_flat]                                 # (N, K+1)
            labels = torch.zeros(N, dtype=torch.long, device=predict.device)    # (N)
            return F.cross_entropy(logits_sel, labels, reduction='mean')        # ()

        raise NotImplementedError(f"Loss type '{self.loss_type}' is not implemented.")
