import numpy as np
import pandas as pd
import polars as pl

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

def count_polars(df: pl.LazyFrame | pl.DataFrame):
    return df.select(pl.len()).collect().item()
