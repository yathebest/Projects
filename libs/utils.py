import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

USER = 'user_id'
ITEM = 'item_id'

TARGET = "rank"
INTERACTIONS_NUM_FEATURES = ['timespent']
INTERACTIONS_CAT_FEATURES = ['place', 'platform', 'agent', ]

USERS_NUM_FEATURES = ['age']
USERS_CAT_FEATURES = ['gender', 'geo']

ITEMS_NUM_FEATURES = ['duration']
ITEMS_CAT_FEATURES = []
ITEM_EMBEDDING = 'embedding'

POSITIVE_INTERACTION_COLUMNS = ['like', 'share', 'bookmark', 'click_on_author', 'open_comments']
NEGATIVE_INTERACTION_COLUMNS = ['dislike']
INTERACTION_COLUMNS = POSITIVE_INTERACTION_COLUMNS + NEGATIVE_INTERACTION_COLUMNS

RANDOM_STATE = 42

def download_files(file_names):
    for file in (file_names):
        hf_hub_download(
            repo_id='deepvk/VK-LSVD', repo_type='dataset',
            filename=file, local_dir='VK-LSVD'
        )

def load_data(train_interactions_files, val_interactions_file, content_embedding_size=64):
    train_interactions = pd.concat([pd.read_parquet(f'VK-LSVD/{file}') for file in train_interactions_files])
    val_interactions = pd.concat([pd.read_parquet(f'VK-LSVD/{file}') for file in val_interactions_file])

    for df in [train_interactions, val_interactions]:
        df[TARGET] = (
                1 * df[POSITIVE_INTERACTION_COLUMNS].any(axis=1)
                - 1 * df[NEGATIVE_INTERACTION_COLUMNS].any(axis=1)
        )
        df.drop(columns=INTERACTION_COLUMNS, inplace=True)

    train_users = train_interactions[USER].unique()
    train_items = train_interactions[ITEM].unique()

    item_emb = np.load('VK-LSVD/metadata/item_embeddings.npz')
    item_ids = item_emb[ITEM]
    item_embeddings = item_emb[ITEM_EMBEDDING]

    mask = np.isin(item_ids, train_items)
    item_ids = item_ids[mask]
    item_embeddings = item_embeddings[mask]
    item_embeddings = item_embeddings[:, :content_embedding_size]

    users_metadata = pd.read_parquet('VK-LSVD/metadata/users_metadata.parquet')
    items_metadata = pd.read_parquet('VK-LSVD/metadata/items_metadata.parquet')

    users_metadata = users_metadata.merge(pd.DataFrame({USER: train_users}), on=USER)
    items_metadata = items_metadata.merge(pd.DataFrame({ITEM: train_items}), on=ITEM)
    items_metadata = items_metadata.merge(pd.DataFrame({ITEM: item_ids, ITEM_EMBEDDING: item_embeddings.tolist()}), on=ITEM)

    for df in [users_metadata, items_metadata]:
        df.drop(columns="train_interactions_rank", inplace=True)

    return (train_interactions, val_interactions), users_metadata, items_metadata

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
