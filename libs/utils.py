import numpy as np
import pandas as pd

from .constants import *

def NDCG(submission: pd.DataFrame, true_reactions: pd.DataFrame):
    """submission is DataFrame[nItems x 2] with columns (item_id, user_ids), where user_ids is List[100]
    true_reactions is DataFrame[nItems x 2] with columns (item_id, user_ids), where user_ids is List"""
    df = submission.merge(
        true_reactions, on=ITEM, how='right', suffixes=["_pred", "_true"]
    ).fillna({USER+'s_pred': ""})

    def calculate_ndcg(row):
        pred_users = row[USER+'s_pred']
        true_users = set(row[USER+'s_true'])
        if not pred_users or not true_users:
            return 0.0

        dcg = 0.0
        for i, user in enumerate(pred_users[:100]):
            if user in true_users:
                dcg += 1 / np.log2(i + 2)

        k = min(len(true_users), 100)
        idcg = sum(1 / np.log2(i + 2) for i in range(k))

        return dcg / idcg if idcg > 0 else 0.0

    ndcg = df.apply(calculate_ndcg, axis=1).mean()
    return ndcg

def DCG(submission: pd.DataFrame, true_reactions: pd.DataFrame):
    df = submission.merge(
        true_reactions, on=ITEM, how='right', suffixes=["_pred", "_true"]
    ).fillna({USER+'s_pred': []})

    def calculate_dcg(row):
        pred_users = row[USER+'s_pred']
        true_users = set(row[USER+'s_true'])

        if not pred_users or not true_users:
            return 0.0

        dcg = 0.0
        for i, user in enumerate(pred_users[:100]):
            if user in true_users:
                dcg += 1 / np.log2(i + 2)

        return dcg

    dcg = df.apply(calculate_dcg, axis=1).mean()
    return dcg
