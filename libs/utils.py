from typing import Literal, Optional

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence

from .constants import *


def NDCG(submission: pd.DataFrame, true_reactions: pd.DataFrame) -> float:
    """
    :param submission is DataFrame[nItems x 2] with columns (ITEM, USER), where USER is List[100]
    :param true_reactions is DataFrame[nItems x 2] with columns (ITEM, USER), where USER is List
    """
    df = submission.merge(
        true_reactions, on=ITEM, how='right', suffixes=["_pred", "_true"]
    ).fillna({USER+'_pred': ""})

    def calculate_ndcg(row):
        pred_users = row[USER+'_pred']
        true_users = set(row[USER+'_true'])

        if isinstance(pred_users, np.ndarray):
            pred_users = pred_users.tolist()
        if isinstance(true_users, np.ndarray):
            true_users = true_users.tolist()

        if not pred_users or not true_users:
            return 0.0

        dcg = 0.0
        for i, user in enumerate(pred_users[:100]):
            if user in true_users:
                dcg += 1 / np.log2(i + 2)

        idcg = (1 / np.log2(np.arange(min(len(true_users), 100)) + 2)).sum()

        return dcg / idcg if idcg > 0 else 0.0

    ndcg = df.apply(calculate_ndcg, axis=1).mean()
    return ndcg

def DCG(submission: pd.DataFrame, true_reactions: pd.DataFrame) -> float:
    """
    :param submission is DataFrame[nItems x 2] with columns (ITEM, USER), where USER is List[100]
    :param true_reactions is DataFrame[nItems x 2] with columns (ITEM, USER), where USER is List
    """
    df = submission.merge(
        true_reactions, on=ITEM, how='right', suffixes=["_pred", "_true"]
    ).fillna({USER+'_pred': ""})

    def calculate_dcg(row):
        pred_users = row[USER+'_pred']
        true_users = set(row[USER+'_true'])

        if isinstance(pred_users, np.ndarray):
            pred_users = pred_users.tolist()
        if isinstance(true_users, np.ndarray):
            true_users = true_users.tolist()

        if not pred_users or not true_users:
            return 0.0

        dcg = 0.0
        for i, user in enumerate(pred_users[:100]):
            if user in true_users:
                dcg += 1 / np.log2(i + 2)

        return dcg

    dcg = df.apply(calculate_dcg, axis=1).mean()
    return dcg

def count_polars(df: pl.LazyFrame | pl.DataFrame | None):
    if df is None:
        return None
    return df.select(pl.len()).collect().item()

def build_embeddings_map(items_df: pl.LazyFrame, items: list | set) -> dict:
    """
    :return: mapping ITEM -> EMBEDDING
    """
    df = items_df.filter(pl.col(ITEM).is_in(list(items))).select([ITEM, EMBEDDING]).collect()
    return {r[0]: np.asarray(r[1], dtype=np.float32) for r in df.rows()}

def build_sequences_from_map(mapping: dict, item_lists: list[list],
                             batch_first: bool = False, padding_side: Literal['left', 'right'] = 'right',
                             device: torch.device | str = 'cpu') -> Tensor:
    """
    :return: Tensor (B, T, D) if batch_first else (T, B, D)
    """
    return pad_sequence([
        torch.from_numpy(np.stack([mapping[it] for it in item_list]))
        for item_list in item_lists
    ], padding_value=0.0, batch_first=batch_first, padding_side=padding_side).to(dtype=torch.float32, device=device)

def build_embedding_sequences(items_df: pl.LazyFrame, item_lists: list[list],
                              batch_first: bool = False, padding_side: Literal['left', 'right'] = 'right',
                              device: torch.device | str = 'cpu') -> Tensor:
    """
    :return: Tensor (B, T, D) if batch_first else (T, B, D)
    """
    mapping = build_embeddings_map(items_df, set().union(*item_lists))
    return build_sequences_from_map(mapping, item_lists, device=device, batch_first=batch_first, padding_side=padding_side)

def build_mask(item_lists: list[list],
               batch_first: bool = False, padding_side: Literal['left', 'right'] = 'right',
               device: torch.device| str = 'cpu'):
    """
    :return: Bool Tensor (B, T) if batch_first else (T, B)
    """
    lengths = torch.tensor([len(l) for l in item_lists])
    max_len = lengths.max().item()
    indices = torch.arange(max_len)

    if padding_side == 'right':
        mask = (indices.unsqueeze(0) < lengths.unsqueeze(1))
    elif padding_side == 'left':
        mask = (indices.unsqueeze(0) >= (max_len - lengths.unsqueeze(1)))
    else:
        raise NotImplementedError()

    mask = mask.to(device=device)

    if not batch_first:
        return mask.t()
    return mask

def count_params(model: Module) -> int:
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def sample_polars(ldf: pl.LazyFrame , n_rows: Optional[int] = None, id_columns: Optional[list[str]] = None):
    if n_rows is None:
        return ldf

    if id_columns is None:
        id_columns = ldf.collect_schema().names()

    lazy_with_hash = ldf.with_columns(
        pl.concat_str(*id_columns).hash(seed=RANDOM_STATE).alias('_row_hash')
    )

    sampled = lazy_with_hash.sort('_row_hash').head(n_rows).drop('_row_hash')

    return sampled