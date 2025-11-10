from typing import Literal

import polars as pl
from torch import nn


class BaseRecurrentModel(nn.Module):
    def __init__(self, trainable: bool = True):
        super().__init__()
        self.trainable = trainable

    def process_data_batch(self, data_batch: pl.DataFrame, items_df: pl.LazyFrame, mode: Literal['train', 'val', 'predict']):
        raise NotImplementedError()
