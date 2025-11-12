import numpy as np
import pandas as pd
import polars as pl
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence

from .constants import *


def NDCG(submission: pd.DataFrame, true_reactions: pd.DataFrame):
    """submission is DataFrame[nItems x 2] with columns (ITEM, USER), where USER is List[100]
    true_reactions is DataFrame[nItems x 2] with columns (ITEM, USER), where USER is List"""
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

def DCG(submission: pd.DataFrame, true_reactions: pd.DataFrame):
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
                             device: torch.device = torch.device('cpu'), batch_first: bool = False) -> Tensor:
    """
    :return: Tensor (T, B, D) or (B, T, D)
    """
    return pad_sequence([
        torch.from_numpy(np.stack([mapping[it] for it in item_list]))
        for item_list in item_lists
    ], padding_value=0.0, batch_first=batch_first).to(dtype=torch.float32, device=device)

def build_embedding_sequences(items_df: pl.LazyFrame, item_lists: list[list],
                              device: torch.device = torch.device('cpu'), batch_first: bool = False) -> Tensor:
    """
    :return: Tensor (T, B, D) or (B, T, D)
    """
    mapping = build_embeddings_map(items_df, set().union(*item_lists))
    return build_sequences_from_map(mapping, item_lists, device=device, batch_first=batch_first)

def build_mask(item_lists: list[list], device: torch.device = torch.device('cpu'), batch_first: bool = False):
    """
    :return: Bool Tensor (T, B) or (B, T)
    """
    lengths = torch.tensor([len(l) for l in item_lists])
    indices = torch.arange(lengths.max().item())
    mask = (indices.unsqueeze(0) < lengths.unsqueeze(1)).transpose(0, 1).to(device=device)
    if batch_first:
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