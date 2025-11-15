from typing import Literal, Union
from numpy.typing import NDArray

import numpy as np
import polars as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor

from .BaseRecurrentModel import BaseRecurrentModel
from .constants import *
from .utils import build_embedding_sequences


class WeightedAvgModel(BaseRecurrentModel):
    """
    :param method: 'add' or 'mul'
    :param temporal_distribution: distribution of weights applied to temporal indices
    :param alpha: for 'add' method - weight of temporal component
    :param temp: for 'exp' temporal_distribution - multiplier in exp
    """

    def __init__(self, method: Literal['add', 'mul'] = 'add',
                 temporal_distribution: Literal['linear', 'exp', 'uni'] = 'linear',
                 alpha: float = 0.5, temp: float = 0.1):
        super().__init__(trainable=False)
        self.method = method
        self.temporal_distribution = temporal_distribution
        self.alpha = alpha
        self.temp = temp

    def process_data_batch(self, data_batch: pl.DataFrame, items_df: pl.LazyFrame,
                           mode: Literal['train', 'val', 'predict'] = 'train') -> Union[NDArray, Tensor]:
        device = next(self.parameters()).device if any(p.numel() for p in self.parameters()) else torch.device('cpu')

        item_lists = data_batch[ITEM].to_list()
        time_index_lists = data_batch[TIME_INDEX].to_list()
        target_lists = data_batch[TARGET].to_list()

        inputs = build_embedding_sequences(items_df, item_lists, batch_first=False, device=device)  # (T, B, D)

        weights = pad_sequence([  # (T, B)
            torch.from_numpy(self._get_weights(row_time, row_target))
            for row_time, row_target in zip(time_index_lists, target_lists)
        ], batch_first=False, padding_value=0.0).to(dtype=torch.float32, device=device)

        out = (weights.unsqueeze(-1) * inputs) # (T, B, D)

        if mode in ['train', 'val']:
            raise NotImplementedError()

        elif mode == 'predict':
            return out.sum(dim=0).detach().cpu().numpy().astype(np.float32)  # (B, D)

        else:
            raise NotImplementedError()

    def _get_weights(self, indices: NDArray, ranks: NDArray) -> NDArray:
        indices = np.asarray(indices)
        ranks = np.asarray(ranks, dtype=float)
        if indices.shape != ranks.shape:
            raise NotImplementedError()

        n = len(indices)
        order = np.argsort(np.argsort(indices)) + 1

        if self.temporal_distribution == 'linear':
            temporal_w = order / order.sum()
        elif self.temporal_distribution == 'exp':
            exp = np.exp(self.temp * order)
            temporal_w = exp / np.sum(exp)
        elif self.temporal_distribution == 'uni':
            temporal_w = np.ones(n) / n
        else:
            raise NotImplementedError()

        mag_sum = ranks.sum()
        if mag_sum == 0:
            magnitude_w = np.ones_like(ranks) / n
        else:
            magnitude_w = ranks / mag_sum

        if self.method == 'mul':
            combined = temporal_w * magnitude_w
        elif self.method == 'add':
            combined = self.alpha * temporal_w + (1.0 - self.alpha) * magnitude_w
        else:
            raise NotImplementedError()

        return combined / combined.sum()