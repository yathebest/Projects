from collections import defaultdict

import numpy as np
import pandas as pd

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
        if not pred_users or not true_users:
            return 0.0

        dcg = 0.0
        for i, user in enumerate(pred_users[:100]):
            if user in true_users:
                dcg += 1 / np.log2(i + 2)

        return dcg

    dcg = df.apply(calculate_dcg, axis=1).mean()
    return dcg


def unconstrained_top_selection(df: pd.DataFrame, top_per_item: int=100) -> pd.DataFrame:
    """:param df index is USER, columns are ITEMs, entries are scores
    :param top_per_item maximum amount of USERs selected for each ITEM
    :returns DataFrame with columns (ITEM, USER), where USER is List[top_per_item]"""
    return df \
        .apply(lambda col: col.nlargest(top_per_item).index.tolist(), axis=0) \
        .melt(var_name=ITEM, value_name=USER) \
        .dropna() \
        .groupby(ITEM)[USER] \
        .apply(list) \
        .reset_index()


def constrained_top_selections(df: pd.DataFrame, top_per_item: int=100, max_per_user: int=101) -> pd.DataFrame:
    """:param df index is USER, columns are ITEMs, entries are scores
    :param top_per_item maximum amount of USERs selected for each ITEM
    :param max_per_user maximum amount of selections per each user
    :returns DataFrame with columns (ITEM, USER), where USER is List[top_per_item]"""
    user_counts = defaultdict(int)
    results = []

    for item in df.columns:
        item_scores = []
        for user in df.index:
            score = df.loc[user, item]
            item_scores.append((-score, user))

        item_scores.sort()

        selected_for_item = []
        for neg_score, user in item_scores:
            if len(selected_for_item) >= top_per_item:
                break
            if user_counts[user] < max_per_user:
                selected_for_item.append(user)
                user_counts[user] += 1
                results.append({
                    USER: user,
                    ITEM: item,
                })

    return pd.DataFrame(results).groupby(ITEM).agg({USER: list})