from typing import Literal

import polars as pl
from torch import nn


class BaseRecurrentModel(nn.Module):
    def __init__(self, trainable: bool = True, padding_side: Literal['left', 'right'] = 'right'):
        super().__init__()
        self.trainable = trainable
        self.padding_side = padding_side

    def process_data_batch(self, data_batch: pl.DataFrame, items_df: pl.LazyFrame, mode: Literal['train', 'val', 'predict']):
        """
        Returns Tensor of shape:
            - mode == 'train'   : (T, B, D) requires_grad == True
            - mode == 'val'     : (T, B, D) requires_grad == False
            - mode == 'predict' : (T, B)    requires_grad == False
        """
        raise NotImplementedError()
