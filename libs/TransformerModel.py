from typing import Literal, Optional, Union

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor

from .BaseRecurrentModel import BaseRecurrentModel
from .constants import *
from .utils import build_embedding_sequences, build_mask


class TransformerModel(BaseRecurrentModel):
    def __init__(self,
                 in_dim: int,
                 out_dim: Optional[int] = None,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 nheads: int = 4,
                 dim_feedforward: Optional[int] = None,
                 dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__(trainable=True)

        self.in_dim = in_dim
        self.out_dim = in_dim if out_dim is None else out_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nheads = nheads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * hidden_dim
        self.max_seq_len = max_seq_len

        if hidden_dim % nheads != 0:
            raise NotImplementedError()

        if self.in_dim != self.hidden_dim:
            self.input_proj = nn.Linear(self.in_dim, self.hidden_dim, bias=False)
        else:
            self.input_proj = nn.Identity()

        self.positional_embedding = nn.Parameter(torch.zeros(self.max_seq_len, self.hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nheads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        if self.out_dim != self.hidden_dim:
            self.proj = nn.Linear(self.hidden_dim, self.out_dim, bias=False)
        else:
            self.proj = nn.Identity()

        nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> Tensor:
        """
        :return: bool Tensor (T, T) additive mask: -inf above diagonal
        """
        mask = torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)
        return mask

    def forward(self, x: Tensor, src_key_padding_mask: Optional[Tensor] = None):
        """
        :param x: Tensor (B, T, I)
        :param src_key_padding_mask: optional bool Tensor (B, T) where True = padding
        :returns: Tensor (B, T, O)
        """
        device = x.device
        if x.dim() != 3:
            raise ValueError("x must be shape (T, B, I)")

        T = x.shape[1]
        if T > self.max_seq_len:
            raise NotImplementedError(f"Sequence length {T} > max_seq_len {self.max_seq_len}")

        x_proj = self.input_proj(x)  # (B, T, H)

        pos = self.positional_embedding[:T].unsqueeze(0)  # (1, T, H)
        x_emb = x_proj + pos  # (B, T, H)

        src_mask = self._causal_mask(T, device)  # (T, T)
        out = self.transformer(x_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)  # (B, T, H)

        out = self.proj(out)  # (B, T, O)

        return out

    def process_data_batch(self,
                           data_batch: pl.DataFrame,
                           items_df: pl.LazyFrame,
                           mode: Literal['train', 'val', 'predict'] = 'train'
                           ) -> Union[NDArray, Tensor]:
        device = next(self.parameters()).device if any(p.numel() for p in self.parameters()) else torch.device('cpu')
        item_lists = data_batch[ITEM].to_list()

        def _run() -> Tensor:
            input = build_embedding_sequences(items_df, item_lists, device, batch_first=True)  # (B, T, I)
            key_padding_mask = ~build_mask(item_lists, device, batch_first=True)  # (B, T) True for padding
            out = self.forward(input, src_key_padding_mask=key_padding_mask)  # (B, T, O)

            return out.permute(1, 0, 2)   # (T, B, O)

        if mode == 'train':
            outputs = _run()  # (T, B, O)
            return outputs

        elif mode in ('val', 'predict'):
            prev_train = self.training
            self.eval()
            with torch.no_grad():
                outputs = _run()  # (T, B, O)
            self.train(prev_train)

            if mode == 'val':
                return outputs  # (T, B, O)
            else:
                lengths = torch.tensor([len(l) for l in item_lists], device=device)  # (B,)
                batch_indices = torch.arange(outputs.shape[1], device=device)  # (B,)
                last = outputs[lengths - 1, batch_indices].detach().cpu().numpy().astype(np.float32)  # (B, O)
                return last
        else:
            raise NotImplementedError()
