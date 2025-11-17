from typing import List, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..constants import *


class NumericalEmbedding(nn.Module):
    """
    :param numerical_features: number of numerical features
    :param dim: dimension of embeddings
    """
    def __init__(self, numerical_features: int, dim: int = 16):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(numerical_features, dim))
        self.biases = nn.Parameter(torch.randn(numerical_features, dim))

    def forward(self, x: Tensor):
        """
        :param x: Tensor (B, Nn)
        :return: Tensor (B, Nn, H)
        """
        return x.unsqueeze(-1) * self.weights + self.biases


class TimeEmbedding(nn.Module):
    """
    :param temporal_features: number of temporal features
    :param dim: dimension of embeddings
    :param reduce: reduction method of cosine and sine embeddings
    """
    def __init__(self, temporal_features: int, dim: int = 16,
                 reduce: Literal['mean', 'sum', 'none', 'linear'] = "sum"):
        super().__init__()
        first_half_dim = dim // 2
        second_half_dim = (dim + 1) // 2

        self.wl = nn.Parameter(torch.randn((temporal_features, first_half_dim)))
        self.bl = nn.Parameter(torch.randn((temporal_features, first_half_dim)))
        self.wp = nn.Parameter(torch.randn((temporal_features, second_half_dim)))
        self.bp = nn.Parameter(torch.randn((temporal_features, second_half_dim)))

        if reduce == 'sum':
            self.reduce_fn = lambda x: x.sum(dim=-1)
        elif reduce == 'mean':
            self.reduce_fn = lambda x: x.mean(dim=-1)
        elif reduce == 'linear':
            self.reduce_fn = nn.Linear(2*dim, dim, bias=False)
        elif reduce == 'none':
            self.reduce_fn = lambda x: x
        else:
            raise NotImplementedError()

    def forward(self, x: Tensor):
        """
        :param x: Tensor (B, Nt)
        :return: Tensor (B, Nt, 2*H) if reduce is 'none' else (B, Nt, H)
        """
        time_sin = torch.sin(self.wl * x + self.bl)  # (B, Nt, H)
        time_cos = torch.cos(x * self.wp + self.bp)  # (B, Nt, H)
        cat = torch.cat([time_sin, time_cos], dim=-1) # (B, Nt, 2*H)
        return self.reduce_fn(cat)  # (B, Nt, H) | (B, Nt, 2*H)


class Tokenizer(nn.Module):
    """
    :param categories: ordered list of number of values that each category contain
    :param numerical_features: number of numerical features
    :param temporal_features: number of temporal features
    :param out_dim: dimension of final embedding
    :param dropout: dropout rate
    """
    def __init__(self, categories: Optional[List[int]] = None,
                 numerical_features: int = 0,
                 temporal_features: int = 0,
                 embedding_dim: int = 16,
                 out_dim: int = 16,
                 dropout: float = 0.0):
        super().__init__()
        self.categorical_features = len(categories)
        self.unique_categories = sum(categories)

        if self.unique_categories > 0:
            categories_offset = F.pad(torch.tensor(list(categories)), (1, 0))
            categories_offset = categories_offset.cumsum(dim=-1)[:-1] + 1  # + pad
            self.register_buffer('categories_offset', categories_offset)
            self.categorical_emb = nn.Embedding(self.unique_categories + 1, embedding_dim, padding_idx=0)  # + pad

        self.numerical_features = numerical_features
        if self.numerical_features > 0:
            self.numerical_emb = NumericalEmbedding(embedding_dim, self.numerical_features)

        self.temporal_features = temporal_features
        if self.temporal_features > 0:
            self.temporal_emb = TimeEmbedding(embedding_dim, temporal_features)

        self.drop = nn.Dropout(dropout)

        self.out_dim = out_dim
        d_out = (self.categorical_features + self.numerical_features + self.temporal_features * 2) * embedding_dim
        if d_out == 0:
            raise NotImplementedError()

        self.proj = nn.Linear(d_out, out_dim, bias=False)


    def forward(self, **features):
        """
        :param features: mapping feature -> Tensor (B, Ni)
        :return: Tensor(B, O)
        """
        tokens = []

        if self.categorical_features > 0:
            categorical = features[CATEGORICAL] + self.categories_offset[:self.categorical_features]
            tokens.append(self.categorical_emb(categorical))  # (B, Nc, H)

        if self.numerical_features > 0:
            tokens.append(self.numerical_emb(features[NUMERICAL]))  # (B, Nn, H)

        if self.temporal_features > 0:
            tokens.append(self.temporal_emb(features[TEMPORAL]))  # (B, Nt, H) | (B, Nt, 2*H)

        tokens = torch.cat(tokens, dim=1)  # (B, N, X*H)
        tokens = tokens.view(tokens.shape[0], -1) # (B, N*X*H)
        tokens = self.drop(tokens)

        return  self.proj(tokens)  # (B, O)
