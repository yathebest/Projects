import os
from collections import defaultdict
from typing import Literal, Optional, Any, Iterable

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
    count_users = defaultdict(int)
    for user_list in submission[USER].iloc:
        for user in user_list:
            count_users[user] += 1

    out_of_max = dict(filter(lambda x: x[1] > MAX_PER_USER, count_users.items()))
    if len(out_of_max) > 0:
        print(f"MAX_PER_USER constraint is not satisfied for {len(out_of_max)} users, "
              f"they will not be considered for metric")

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
        pred_users = [u for u in pred_users if u not in out_of_max]
        for i, user in enumerate(pred_users[:TOP_PER_ITEM]):
            if user in true_users:
                dcg += 1 / np.log2(i + 2)

        idcg = (1 / np.log2(np.arange(min(len(true_users), TOP_PER_ITEM)) + 2)).sum()

        return dcg / idcg if idcg > 0 else 0.0

    ndcg = df.apply(calculate_ndcg, axis=1).mean()
    return ndcg

def DCG(submission: pd.DataFrame, true_reactions: pd.DataFrame) -> float:
    """
    :param submission is DataFrame[nItems x 2] with columns (ITEM, USER), where USER is List[100]
    :param true_reactions is DataFrame[nItems x 2] with columns (ITEM, USER), where USER is List
    """
    count_users = defaultdict(int)
    for user_list in submission[USER].iloc:
        for user in user_list:
            count_users[user] += 1

    out_of_max = dict(filter(lambda x: x[1] > MAX_PER_USER, count_users.items()))
    if len(out_of_max) > 0:
        print(f"MAX_PER_USER constraint is not satisfied for {len(out_of_max)} users, "
              f"they will not be considered for metric")

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
        pred_users = [u for u in pred_users if u not in out_of_max]
        for i, user in enumerate(pred_users[:TOP_PER_ITEM]):
            if user in true_users:
                dcg += 1 / np.log2(i + 2)

        return dcg

    dcg = df.apply(calculate_dcg, axis=1).mean()
    return dcg

def build_multimap(ldf: pl.LazyFrame, keys: Iterable,
                   key_column: str = ITEM, value_columns: Iterable[str] = (EMBEDDING,)):
    """
    :param ldf: LazyFrame from which items will be mapped
    :param keys: keys to map
    :param key_column: column to select as key from ldf
    :param value_columns: columns to select as values from ldf
    :return: mapping key -> value_column -> value
    """
    cols = list(value_columns)
    df = ldf.filter(pl.col(key_column).is_in(list(keys))).select([key_column, *cols]).collect()
    return {r[0]: dict(zip(cols, r[1:])) for r in df.rows()}

def build_map(ldf: pl.LazyFrame, keys: Iterable,
              key_column: str = ITEM, value_column: str = EMBEDDING) -> dict[Any, Any]:
    """
    :param ldf: LazyFrame from which items will be mapped
    :param keys: keys to map
    :param key_column: column to select as key from ldf
    :param value_column: column to select as value from ldf
    :return: mapping key -> value
    """

    df = ldf.filter(pl.col(key_column).is_in(list(keys))).select([key_column, value_column]).collect()
    return {r[0]: r[1] for r in df.rows()}

def build_sequences_from_map(mapping: dict, key_lists: list[list[Any]],
                             batch_first: bool = False, padding_side: Literal['left', 'right'] = 'right',
                             device: torch.device | str = 'cpu') -> Tensor:
    """
    :return: Tensor (B, T, *) if batch_first else (T, B, *)
    """
    return pad_sequence([
        torch.tensor([mapping[it] for it in item_list])
        for item_list in key_lists
    ], padding_value=0, batch_first=batch_first, padding_side=padding_side).to(device=device)

def build_sequences(items_df: pl.LazyFrame, key_lists: list[list[Any]],
                    key_column: str = ITEM, value_column: str = EMBEDDING,
                    batch_first: bool = False, padding_side: Literal['left', 'right'] = 'right',
                    device: torch.device | str = 'cpu') -> Tensor:
    """
    :return: Tensor (B, T, D) if batch_first else (T, B, D)
    """
    mapping = build_map(items_df, keys=set().union(*key_lists), key_column=key_column, value_column=value_column)
    return build_sequences_from_map(mapping, key_lists, device=device, batch_first=batch_first, padding_side=padding_side)

def build_mask(lists: list[list[Any]],
               batch_first: bool = False, padding_side: Literal['left', 'right'] = 'right',
               device: torch.device| str = 'cpu'):
    """
    :return: Bool Tensor (B, T) if batch_first else (T, B)
    """
    lengths = torch.tensor([len(l) for l in lists])
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
    """
    :param model: nn.Module
    :return: number of trainable parameters
    """
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def count_polars(df: pl.LazyFrame | pl.DataFrame) -> int:
    length = df.select(pl.len())

    if isinstance(length, pl.LazyFrame):
        length = length.collect()

    return length.item()

def sample_polars(ldf: pl.LazyFrame , n_rows: Optional[int] = None, id_columns: Optional[list[str]] = None):
    """
    :param ldf: polars LazyDataFrame
    :param n_rows: number of rows to sample
    :param id_columns: optional columns, containing unique id for each row
    :return: polars LazyFrame with 'n_rows' rows sampled randomly
    """
    if n_rows is None:
        return ldf

    if id_columns is None:
        id_columns = ldf.collect_schema().names()

    lazy_with_hash = ldf.with_columns(
        pl.concat_str(*id_columns).hash(seed=RANDOM_STATE).alias('_row_hash')
    )

    sampled = lazy_with_hash.sort('_row_hash').head(n_rows).drop('_row_hash')

    return sampled

def clear_cache(dir: str = CACHE_DIR, files_to_keep: list[str] = ()):
    for item_name in os.listdir(dir):
        if item_name not in files_to_keep:
            try:
                os.remove(os.path.join(dir, item_name))
            except OSError as e:
                print(f"Error deleting {item_name}: {e}")