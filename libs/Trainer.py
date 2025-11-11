from math import ceil
from typing import Optional, Literal

import numpy as np
import polars as pl
from torch import Tensor
from tqdm.auto import tqdm, trange
import torch
import torch.nn.functional as F
from torch.optim import Optimizer, AdamW
from copy import deepcopy

from .BaseRecurrentModel import BaseRecurrentModel
from .utils import count_polars, build_embedding_sequences
from .constants import *


class Trainer:
    """
    :param model: Recurrent model to predict embeddings
    :param train_interactions: LazyFrame containing train interactions with columns [USER, ITEM, TARGET, TIME_INDEX]
    :param predict_items: LazyFrame containing items to predict users for with columns [ITEM]
    :param items_metadata: LazyFrame containing all items metadata with columns [ITEM, EMBEDDING]
    :param num_recent_videos: number of known the most recent interactions to keep for each user
    :param val_ratio: ratio for temporal splitting interactions into train/val
    """
    def __init__(self, model: BaseRecurrentModel,
                 train_interactions: pl.LazyFrame, predict_items: pl.LazyFrame,
                 items_metadata: pl.LazyFrame,
                 num_recent_videos: int = 500, val_ratio: float = 0.0,
                 loss_type: Literal['mse', 'cos'] = 'mse',
                 device: torch.device = torch.device('cpu')
                 ):
        self.model = model.to(device)
        self.test = predict_items
        self.items_df = items_metadata
        self.val_ratio = val_ratio
        self.loss_type = loss_type
        self.device = device
        self.history = None

        def _agg(interactions: pl.LazyFrame):
            return interactions \
                .sort(by=[pl.col(USER), pl.col(TIME_INDEX)], descending=[False, True]) \
                .group_by(USER) \
                .agg([pl.col(ITEM).slice(0, num_recent_videos).alias(ITEM),
                      pl.col(TIME_INDEX).slice(0, num_recent_videos).alias(TIME_INDEX),
                      pl.col(TARGET).slice(0, num_recent_videos).alias(TARGET)])

        self.data = _agg(train_interactions)
        if val_ratio <= 0.0:
            print(f"Val ratio is too small, train/val split is not used")
            self.train = self.data
            self.val = None
        else:
            print(f"Temporal train/val split with ratio val_ratio {val_ratio}")
            split_time = train_interactions.select(pl.col(TIME_INDEX).quantile(1-val_ratio)).collect().item()

            # higher TIME_INDEX -> newer
            self.train = _agg(train_interactions.filter(pl.col(TIME_INDEX) < split_time))
            self.val = _agg(train_interactions.filter(pl.col(TIME_INDEX) >= split_time))

        print("Count")
        self.data_count = count_polars(self.data)
        self.train_count = count_polars(self.train)
        self.val_count = count_polars(self.val)
        self.test_count = count_polars(self.test)
        print(f"All: {self.data_count} Train: {self.train_count} Val: {self.val_count} Test: {self.test_count}")

    def _build_target_with_mask(self, item_lists: list[list]) -> tuple[Tensor, Tensor]:
        target_tensor = build_embedding_sequences(self.items_df, item_lists, self.device)

        lengths = torch.tensor([len(l) for l in item_lists])
        indices = torch.arange(lengths.max().item())
        mask = (indices.unsqueeze(0) < lengths.unsqueeze(1)).transpose(0, 1).to(device=self.device)

        return target_tensor, mask

    def _compute_loss(self, predict: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        if self.loss_type == 'mse':
            diff = predict - target  # (L, B, D)
            sq = diff.pow(2).sum(dim=2)  # (L, B)
            masked_sq_sum = sq[mask].sum()
            emb_dim = predict.shape[2]
            denom = float(mask.sum().item() * emb_dim)
            return masked_sq_sum / denom

        elif self.loss_type == 'cos':
            out_n = F.normalize(predict, p=2, dim=2)
            tgt_n = F.normalize(target, p=2, dim=2)
            cos = (out_n * tgt_n).sum(dim=2)  # (L, B)
            cos = torch.clamp(cos, -1.0, 1.0)
            per_pos = (1.0 - cos)[mask]
            return per_pos.mean()

        else:
            raise NotImplementedError()

    def train_epoch(self, optimizer: Optimizer, users_batch_size: int = 4096, epoch: Optional[int] = None, verbose=True) -> float:
        self.model.train()

        total_batches = max(1, ceil(self.train_count / users_batch_size)) if self.train_count else 1
        running_loss = 0.0
        processed_batches = 0

        dataiter = self.train.collect_batches(chunk_size=users_batch_size)
        if verbose:
            bar = tqdm(total=total_batches, desc=f"Train Epoch: {epoch}" if epoch is not None else "Train Epoch")
        else:
            bar = None

        for data_batch in dataiter:

            predict = self.model.process_data_batch(data_batch, items_df=self.items_df, mode='train')
            predict = predict[:-1]

            target, mask = self._build_target_with_mask(data_batch[ITEM].to_list())
            target, mask = target[1:], mask[1:]

            batch_loss = self._compute_loss(predict, target, mask)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            running_loss += float(batch_loss.item())
            processed_batches += 1

            avg_loss = running_loss / processed_batches if processed_batches else 0.0
            if bar:
                bar.set_description(f"Train Epoch: {epoch} Loss: {avg_loss:.4f}" if epoch is not None else f"Train Loss: {avg_loss:.4f}")
                bar.update(1)

        return (running_loss / processed_batches) if processed_batches else 0.0

    def val_epoch(self, users_batch_size: int = 4096, epoch: Optional[int] = None, verbose=True) -> float:
        if self.val is None:
            return 0.0

        self.model.eval()

        total_batches = max(1, ceil(self.val_count / users_batch_size)) if self.val_count else 1
        running_loss = 0.0
        processed_batches = 0

        dataiter = self.val.collect_batches(chunk_size=users_batch_size)
        if verbose:
            bar = tqdm(total=total_batches,desc=f"Val Epoch: {epoch}" if epoch is not None else "Val Epoch")
        else:
            bar = None
        with torch.no_grad():
            for data_batch in dataiter:

                predict = self.model.process_data_batch(data_batch, items_df=self.items_df, mode='val')
                predict = predict[:-1]

                target, mask = self._build_target_with_mask(data_batch[ITEM].to_list())
                target, mask = target[1:], mask[1:]

                batch_loss = self._compute_loss(predict, target, mask)

                running_loss += float(batch_loss.cpu().item())
                processed_batches += 1

                avg_loss = running_loss / processed_batches if processed_batches else 0.0
                if bar:
                    bar.set_description(f"Val Epoch: {epoch} Loss: {avg_loss:.4f}" if epoch is not None else f"Val Loss: {avg_loss:.4f}")
                    bar.update(1)

        return (running_loss / processed_batches) if processed_batches else 0.0

    def fit(self, epochs: int = 5, users_batch_size: int = 4096, lr: float = 1e-3, weight_decay: float = 1e-2, verbose=True):
        if not self.model.trainable:
            print("Model is not trainable")
            return

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.history: dict[str, list] = {'train_loss': [], 'val_loss': []}

        best = (-1, float('inf'), deepcopy(self.model))

        if not verbose:
            bar = tqdm(total=epochs)
        else:
            bar = None
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(optimizer, users_batch_size=users_batch_size, epoch=epoch, verbose=verbose)
            self.history['train_loss'].append(train_loss)

            if self.val is not None:
                val_loss = self.val_epoch(users_batch_size=users_batch_size, epoch=epoch, verbose=verbose)
                if val_loss < best[1]:
                    best = (epoch, val_loss, deepcopy(self.model.state_dict()))

                self.history['val_loss'].append(val_loss)
            else:
                val_loss = None
                if train_loss < best[1]:
                    best = (epoch, val_loss, deepcopy(self.model.state_dict()))
                self.history['val_loss'].append(None)

            if bar:
                bar.set_description(f"Epoch: {epoch} Train Loss: {train_loss:.4f}" + f" Val Loss: {val_loss:.4f}" if val_loss else "")
                bar.update(1)

        print(f"Completed! Best Loss {best[1]:.4f} at Epoch {best[0]}")
        self.model.load_state_dict(best[2])
