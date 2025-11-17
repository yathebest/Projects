from typing import Literal, Optional, Dict
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from numpy.typing import NDArray
from torch import Tensor

from .BaseRecurrentModel import BaseRecurrentModel
from .Tokenizer import Tokenizer
from ..constants import *
from ..utils import build_mask, build_multimap, build_sequences


class TransformerModel(BaseRecurrentModel):
    def __init__(self,
                 item_embedding_dim: int,
                 user_tokenizer: Optional[Tokenizer] = None,
                 author_embedding: Optional[nn.Embedding] = None,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 nheads: int = 4,
                 dim_feedforward: Optional[int] = None,
                 dropout: float = 0.1,
                 trainable_position: bool = False,
                 max_seq_len: int = 512,
                 use_multi_target: bool = False):
        super().__init__(trainable=True)

        self.in_dim = item_embedding_dim  # I
        self.out_dim = item_embedding_dim  # O
        self.hidden_dim = hidden_dim  # H
        self.num_layers = num_layers
        self.nheads = nheads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward if dim_feedforward is not None else 4 * hidden_dim
        self.max_seq_len = max_seq_len
        self.use_multi_target = use_multi_target

        if hidden_dim % nheads != 0:
            raise NotImplementedError()

        self.user_tokenizer = user_tokenizer
        if self.user_tokenizer is not None:
            if self.hidden_dim != user_tokenizer.out_dim:
                self.user_proj = nn.Linear(user_tokenizer.out_dim, self.hidden_dim)
            else:
                self.user_proj = nn.Identity()

        self.author_embedding = author_embedding
        if self.author_embedding is not None:
            self.author_vocab_size = author_embedding.embedding_dim
            combined_input_dim = self.in_dim + self.author_vocab_size
        else:
            combined_input_dim = self.in_dim

        self.input_proj = nn.Linear(combined_input_dim, self.hidden_dim, bias=False)

        if trainable_position:
            self.positional_embedding = nn.Embedding(max_seq_len, hidden_dim)
            nn.init.normal_(self.positional_embedding.weight, mean=0.0, std=0.02)
        else:
            self.positional_embedding = self._create_sinusoidal_embeddings(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.nheads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.embedding_head = nn.Linear(self.hidden_dim, self.out_dim, bias=False)

        if self.use_multi_target:
            self.author_head = nn.Linear(self.hidden_dim, self.author_vocab_size)
            self.rank_head = nn.Linear(self.hidden_dim, 1)
        else:
            self.proj = nn.Linear(self.hidden_dim, self.out_dim, bias=False)

    @staticmethod
    def _create_sinusoidal_embeddings(max_len: int, dim: int) -> nn.Embedding:
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return nn.Embedding.from_pretrained(pe, freeze=True)

    @staticmethod
    def _causal_mask(size: int, device: torch.device | str = 'cpu') -> Tensor:
        mask = torch.triu(torch.ones((size, size), dtype=torch.bool, device=device), diagonal=1)
        return mask

    def forward(self,
                x: Tensor,
                author_sequence: Optional[Tensor] = None,
                user_tokenizer_xs: Optional[dict[str, Tensor]] = None,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        :param x: Tensor (B, T, I) - item embeddings
        :param user_tokenizer_xs: mapping x_name -> Tensor (B, Ni), where Ni - number of features of each type (e.g. categorical)
        :param author_sequence: Tensor (B, T) - author IDs
        :param src_mask: optional bool Tensor (T, T)
        :param src_key_padding_mask: optional bool Tensor (B, T) where True = padding
        :returns: Dict with predictions for each target
        """
        device = x.device
        T = x.shape[1]

        if T > self.max_seq_len:
            raise NotImplementedError()

        if self.author_embedding is not None:
            author_emb = self.author_embedding(author_sequence)            # (B, T, A)
            combined_input = torch.cat([x, author_emb], dim=-1)     # (B, T, I+A)
        else:
            combined_input = x                                             # (B, T, I)

        x_proj = self.input_proj(combined_input)                           # (B, T, H)

        if self.user_tokenizer is not None:
            user_emb = self.user_tokenizer(**user_tokenizer_xs)     # (B, Hu)
            user_proj = self.user_proj(user_emb).unsqueeze(1)       # (B, 1, H)
            user_bias = user_proj.expand(-1, T, -1)                 # (B, T, H)
            x_proj = x_proj + user_bias                             # (B, T, H)

        position_ids = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        pos_emb = self.positional_embedding(position_ids)           # (1, T, H)

        if src_key_padding_mask is not None:
            non_padding_mask = (~src_key_padding_mask).unsqueeze(-1)    # (B, T, 1)
            x_emb = x_proj + (pos_emb * non_padding_mask)               # (B, T, H)
        else:
            x_emb = x_proj + pos_emb                                    # (B, T, H)

        transformer_out = self.transformer(x_emb, mask=src_mask, src_key_padding_mask=src_key_padding_mask)  # (B, T, H)

        outputs = {EMBEDDING: self.proj(transformer_out)}   # (B, T, O)

        if self.use_multi_target:
            outputs[AUTHOR] = self.author_head(transformer_out)             # (B, T, A)
            outputs[TARGET] = self.rank_head(transformer_out).squeeze(-1)   # (B, T)

        return outputs

    def process_data_batch(self, batch: pl.DataFrame,
                           items_df: pl.LazyFrame, users_df: pl.LazyFrame,
                           mode: Literal['train', 'val', 'predict'] = 'train') -> dict[str, Tensor] | NDArray:
        device = self.device

        item_lists = batch[ITEM].to_list()

        def _run() -> Dict[str, Tensor]:
            user_metadata = None
            if self.user_tokenizer is not None:
                users = batch[USER].to_list()
                user_map = build_multimap(users_df, users, key_column=USER, value_columns=set().union(*USER_FEATURES.values()))

                user_metadata = {
                    feature_type: torch.tensor([
                        [user_map[u][feat] for feat in features]
                        for u in users
                    ])
                    for feature_type, features in USER_FEATURES.items() if len(features) > 0
                }

            author_seq = None
            if self.author_embedding is not None:
                author_seq = build_sequences(items_df, item_lists, key_column=ITEM, value_column=AUTHOR,
                                             batch_first=True, device=device)

            item_seq = build_sequences(items_df, item_lists, key_column=ITEM, value_column=EMBEDDING,
                                       batch_first=True, device=device)

            key_padding_mask = ~build_mask(item_lists, batch_first=True, device=device)     # (B, T)
            src_mask = self._causal_mask(item_seq.shape[1], device=device)                  # (T, T)

            out = self.forward(
                item_seq,
                author_seq,
                user_metadata,
                src_mask=src_mask,
                src_key_padding_mask=key_padding_mask
            )  # mapping out_name -> (B, T, *)

            return {
                k: v.transpose(1, 0)  # (T, B, *)
                for k, v in out.items()
            }

        if mode == 'train':
            outputs = _run()
            return outputs

        elif mode in ('val', 'predict'):
            prev_train = self.training
            self.eval()
            with torch.no_grad():
                outputs = _run()
            self.train(prev_train)

            if mode == 'val':
                return outputs

            else:
                item_pred = outputs[EMBEDDING]  # (T, B, O)
                lasts = torch.tensor([len(l)-1 for l in item_lists], device=device, dtype=torch.long)  # (B)
                batch_indices = torch.arange(len(item_lists), device=device, dtype=torch.long)  # (B)

                embeddings = item_pred[lasts, batch_indices]  # (B, O)
                return embeddings.detach().cpu().numpy().astype(np.float32)

        else:
            raise NotImplementedError()
