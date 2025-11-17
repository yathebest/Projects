from typing import Literal
from numpy.typing import NDArray

import polars as pl
import torch
from torch import Tensor
from torch import nn


class BaseRecurrentModel(nn.Module):
    def __init__(self, trainable: bool = True, padding_side: Literal['left', 'right'] = 'right'):
        super().__init__()
        self.trainable = trainable
        self.padding_side = padding_side

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device if any(p.numel() for p in self.parameters()) else torch.device('cpu')

    def process_data_batch(self, batch: pl.DataFrame,
                           items_df: pl.LazyFrame, users_df: pl.LazyFrame,
                           mode: Literal['train', 'val', 'predict']) -> dict[str, Tensor] | NDArray:
        """
        Returns Tensor of shape:
            - mode == 'train'   : mapping target -> (T, B, D) requires_grad == True
            - mode == 'val'     : mapping target -> (T, B, D) requires_grad == False
            - mode == 'predict' : (T, B) requires_grad == False
        """
        raise NotImplementedError()
