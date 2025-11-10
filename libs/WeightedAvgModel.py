from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .constants import *


class WeightedAvgModel:
    """
    :param method: 'mul' or 'add'
    :param temporal_distribution: distribution of weights applied to temporal indices
    :param alpha: for 'add' method - weight of temporal component
    :param temp: for 'exp' temporal_distribution - multiplier in exp
    """

    def __init__(self, method: Literal['mul', 'add'] = 'mul',
                 temporal_distribution: Literal['linear', 'exp', 'uni'] = 'linear',
                 alpha: float = 0.5, temp: float = 0.1):
        self.method = method
        self.temporal_distribution = temporal_distribution
        self.alpha = alpha
        self.temp = temp

    def process_data_batch(self, data_batch: pl.DataFrame, items_df: pl.LazyFrame, items_batch_size):
        users_batch_size = len(data_batch)
        item_lists = data_batch[ITEM].to_list()  # list of lists
        time_index_lists = data_batch[TIME_INDEX].to_list()
        target_lists = data_batch[TARGET].to_list()

        seen = set()
        data_items_list = []
        user_item_weights = []
        for row_items, row_time, row_target in zip(item_lists, time_index_lists, target_lists):
            weights = self._get_weights(row_time, row_target)
            items_weights = []
            for it, w in zip(row_items, weights):
                if it not in seen:
                    seen.add(it)
                    data_items_list.append(it)
                items_weights.append((it, float(w)))
            user_item_weights.append(items_weights)

        embeddings = None
        for start in range(0, len(data_items_list), items_batch_size):
            chunk = data_items_list[start:start + items_batch_size]

            items_batch_df = items_df.filter(pl.col(ITEM).is_in(chunk)).select([ITEM, EMBEDDING]).collect()
            chunk_map = {r[0]: np.asarray(r[1], dtype=np.float64) for r in items_batch_df.rows()}

            if embeddings is None:
                d = next(iter(chunk_map.values())).shape[0]
                embeddings = np.zeros((users_batch_size, d), dtype=np.float64)

            for ui, items_weights in enumerate(user_item_weights):
                acc = embeddings[ui]
                for it, w in items_weights:
                    emb = chunk_map.get(it)
                    if emb is not None:
                        acc += w * emb
        return embeddings

    def _get_weights(self, indices: NDArray, ranks: NDArray) -> NDArray:
        """
        :param indices: array-like, temporal indices (only order matters)
        :param ranks: array-like, ranks magnitude (only magnitude matters)
        """
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