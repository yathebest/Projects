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

from .BaseRecurrentModel import BaseRecurrentModel
from .constants import *
from .utils import count_polars, build_mask, build_embeddings_map, build_sequences_from_map


class Trainer:
    """
    :param model: Recurrent model to predict embeddings
    :param train_interactions: LazyFrame containing train interactions with columns [USER, ITEM, TARGET, TIME_INDEX]
    :param val_interactions: LazyFrame containing val interactions with columns [USER, ITEM, TARGET, TIME_INDEX]
    :param items_metadata: LazyFrame containing all items metadata with columns [ITEM, EMBEDDING]
    :param loss_type: type of loss function to use
    :param loss_margin: used for margin-based losses (with negative samples)
    :param negative_loss_pooling: used to pool negative loss from K to 1
    :param negative_ratio: used to sample negatives. If provided margin-based loss will be used
    :param num_recent_videos: number of known the most recent interactions to keep for each user
    """
    def __init__(self, model: BaseRecurrentModel,
                 train_interactions: pl.LazyFrame, val_interactions: pl.LazyFrame,
                 items_metadata: pl.LazyFrame,
                 num_recent_videos: int = 500,
                 loss_type: Literal['mse', 'cos'] = 'mse', loss_margin: float = 0.0,
                 negative_ratio: int = 0,
                 negative_loss_pooling: Literal['max', 'mean', 'min'] = 'mean',
                 device: torch.device | str = 'cpu',
                 ):
        self.model = model.to(device)
        self.optimizer = None
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
        :param predict: (T, B, D)
        :param target: (T, B, D)
        :param mask: (T, B) boolean
        :param negative: optional (T, B, K, D) K negative samples per position.
        """

        def pool_negative(negative_loss_all: Tensor):
            fn = getattr(torch, self.negative_loss_pooling)
            return fn(negative_loss_all, dim=2)

        if self.loss_type == 'mse':  # MSELoss | MarginMSELoss
            diff = predict - target  # (T, B, D)
            sq_pos = diff.pow(2).sum(dim=2)  # (T, B)    to minimize

            if negative is None:  # MSELoss
                masked_sq_sum = sq_pos[mask].sum()
                emb_dim = predict.shape[2]
                denom = float(mask.sum().item() * emb_dim) if mask.sum().item() > 0 else 1.0
                return masked_sq_sum / denom

            else:  # MarginMSELoss
                sq_neg_all = (predict.unsqueeze(2) - negative).pow(2).sum(dim=3)  # (T, B, K, D) -> (T, B, K)
                sq_neg = -pool_negative(-sq_neg_all) # (T, B)    to maximize

                hinge = F.relu(self.loss_margin + sq_pos - sq_neg)  # (T, B)
                return hinge[mask].mean()

        elif self.loss_type == 'cos':  # CosineEmbeddingLoss | MarginCosineEmbeddingLoss
            out_n = F.normalize(predict, p=2, dim=2)  # (T, B, D)
            tgt_n = F.normalize(target, p=2, dim=2)  # (T, B, D)
            cos_pos = (out_n * tgt_n).sum(dim=2)  # (T, B)    to maximize
            cos_pos = torch.clamp(cos_pos, -1.0, 1.0)

            if negative is None:  # CosineEmbeddingLoss
                per_pos = (1.0 - cos_pos)[mask]
                return per_pos.mean()

            else:
                neg_n = negative.view(-1, negative.shape[2], negative.shape[3])  # (T, B, K, D) -> (TxB, K, D)
                neg_n = F.normalize(neg_n, p=2, dim=2).view(negative.shape)  # (TxB, K, D) -> (T, B, K, D)
                out_expand = out_n.unsqueeze(2)  # (T, B, 1, D)
                cos_neg_all = (out_expand * neg_n).sum(dim=3)  # (T, B, K)
                cos_neg = pool_negative(cos_neg_all)  # (T, B)    to minimize
                cos_neg = torch.clamp(cos_neg, -1.0, 1.0)
                hinge = F.relu(self.loss_margin + cos_neg - cos_pos)
                return hinge[mask].mean()

        else:
            raise NotImplementedError()

    def _ensure_optimizer(self, **params) -> torch.optim.Optimizer:
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), **params)

        self.optimizer.zero_grad()
        return self.optimizer

    def load_checkpoint(self, path: str = "./models/model.pt"):
        chkpt = torch.load(path)
        self.model.load_state_dict(chkpt["model_dict"])
        self._ensure_optimizer().load_state_dict(chkpt["optimizer_dict"])
        self.start_epoch = chkpt["epoch"] + 1
        self.history = chkpt["history"]

    def train_epoch(self, optimizer: Optimizer, users_batch_size: int = 4096, epoch: Optional[int] = None, verbose=True) -> float:
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

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            running_loss += float(batch_loss.item())
            processed_batches += 1

            avg_loss = running_loss / processed_batches if processed_batches else 0.0
            bar.set_description(f"Train Epoch: {epoch} Loss: {avg_loss:.4f}" if epoch is not None else f"Train Loss: {avg_loss:.4f}")

        return (running_loss / processed_batches) if processed_batches else 0.0

    def val_epoch(self, users_batch_size: int = 4096, epoch: Optional[int] = None, verbose=True) -> float:
        if self.val is None:
            return 0.0

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

    def fit(self, epochs: int = 5, users_batch_size: int = 4096,
            lr: float = 1e-3, weight_decay: float = 1e-2,
            save: bool = True, verbose: bool = True):
        if not self.model.trainable:
            print("Model is not trainable")
            return

        self._ensure_optimizer(lr=lr, weight_decay=weight_decay)
        self.history: dict[str, list] = {'train_loss': [], 'val_loss': []}
        best = (-1, float('inf'), deepcopy(self.model.state_dict()))

        try:
            with trange(self.start_epoch, epochs + 1, disable=verbose) as bar:
                for epoch in bar:
                    train_loss = self.train_epoch(self.optimizer, users_batch_size=users_batch_size, epoch=epoch, verbose=verbose)
                    self.history['train_loss'].append(train_loss)

                    if self.val is not None:
                        val_loss = self.val_epoch(users_batch_size=2*users_batch_size, epoch=epoch, verbose=verbose)
                        candidate_loss = val_loss
                    else:
                        val_loss = None
                        candidate_loss = train_loss

                    self.history['val_loss'].append(val_loss)
                    if candidate_loss < best[1]:
                        best = (epoch, candidate_loss, deepcopy(self.model.state_dict()))

                        if save:
                            if not os.path.exists("./models"):
                                os.mkdir("./models")
                            torch.save({
                                "model_dict": self.model.state_dict(),
                                "optimizer_dict": self.optimizer.state_dict(),
                                "epoch": epoch,
                                "history": self.history,
                            }, "./models/model.pt")

                    bar.set_description(f"Epoch: {epoch} Train Loss: {train_loss:.4f}" +
                                        (f" Val Loss: {val_loss:.4f}" if val_loss else "") +
                                        f"Best Epoch: {best[0]}")

            print(f"Completed! Best Loss {best[1]:.4f} at Epoch {best[0]}")
            self.model.load_state_dict(best[2])

        except Exception as e:
            print(f"Exception raised during fit: {e}\nTaking current best model from Epoch {best[0]} with Loss {best[1]:.4f}")
            self.model.load_state_dict(best[2])
            raise
