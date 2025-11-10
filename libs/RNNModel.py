from typing import Literal, Optional, Tuple, Union
from numpy.typing import NDArray

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch import Tensor

from .BaseRecurrentModel import BaseRecurrentModel
from .constants import *
from .utils import build_embedding_sequences


class RNNModel(BaseRecurrentModel):
    def __init__(self,
                 item_embeddings_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 1,
                 dropout: float = 0.0,
                 layer: Literal['lstm', 'gru', 'rnn'] = 'lstm'):
        super().__init__(trainable=True)

        self.in_dim = item_embeddings_dim
        self.out_dim = item_embeddings_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_type = layer
        self.dropout = dropout

        if layer == 'lstm':
            rnn_cls = nn.LSTM
        elif layer == 'gru':
            rnn_cls = nn.GRU
        elif layer == 'rnn':
            rnn_cls = nn.RNN
        else:
            raise NotImplementedError()

        self.rnn = rnn_cls(
            self.in_dim, self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            bidirectional=False
        )

        if self.out_dim != self.hidden_dim:
            self.proj = nn.Linear(self.hidden_dim, self.out_dim, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x: Tensor, hidden: Optional[Tuple[Tensor, Tensor] | Tensor] = None, return_hidden: bool = False):
        if hidden:
            out, hid = self.rnn(x, hidden)
        else:
            out, hid = self.rnn(x)
        out = self.proj(out)

        return (out, hid) if return_hidden else out

    def process_data_batch(self, data_batch: pl.DataFrame, items_df: pl.LazyFrame,
                           mode: Literal['train', 'val', 'predict'] = 'train'
                           ) -> Union[NDArray, Tensor]:
        device = next(self.parameters()).device if any(p.numel() for p in self.parameters()) else torch.device('cpu')
        item_lists = data_batch[ITEM].to_list()

        def _run() -> Tensor:
            input = build_embedding_sequences(items_df, item_lists, device)
            return self.forward(input, return_hidden=False)

        if mode == 'train':
            outputs = _run()  # (L, B, D)
            return outputs

        elif mode == 'val' or mode == 'predict':
            prev_train = self.training
            self.eval()
            with torch.no_grad():
                outputs = _run()  # (L, B, D)
            self.train(prev_train)

            if mode == 'val':
                return outputs  # (L, B, D)
            else:
                return outputs[-1].detach().cpu().numpy().astype(np.float32) # (B, D)

        else:
            raise NotImplementedError()

