import os
from functools import reduce

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

from .constants import *

class Loader:
    def __init__(self, subsample_name: str, content_embedding_size: int=64):
        self.subsample_name = subsample_name
        self.content_embedding_size = content_embedding_size

    def _get_files(self):
        train_interactions_files = [f'subsamples/{self.subsample_name}/train/week_{i:02}.parquet' for i in range(25)]
        val_interactions_files = [f'subsamples/{self.subsample_name}/validation/week_25.parquet']
        metadata_files = ['metadata/users_metadata.parquet', 'metadata/items_metadata.parquet', 'metadata/item_embeddings.npz']
        return (train_interactions_files, val_interactions_files), metadata_files

    def _download_files(self, files: list=None):
        if files is None:
            (t, v), m = self._get_files()
            files = t+v+m

        for file in files:
            hf_hub_download(
                repo_id='deepvk/VK-LSVD', repo_type='dataset',
                filename=file, local_dir=DATA_DIR
            )

    def _load_data(self):
        (train_interactions_files, val_interactions_file), metadata_files = self._get_files()
        files = train_interactions_files + val_interactions_file + metadata_files
        files = list(filter(lambda path: not os.path.exists(f'{DATA_DIR}/{path}'), files))
        if len(files) != 0:
            print(f"{len(files)} files not found, downloading...")
            self._download_files(files)

        train_interactions = pd.concat([pd.read_parquet(f'{DATA_DIR}/{file}') for file in train_interactions_files])
        val_interactions = pd.concat([pd.read_parquet(f'{DATA_DIR}/{file}') for file in val_interactions_file])

        users_metadata = pd.read_parquet(f'{DATA_DIR}/metadata/users_metadata.parquet')
        items_metadata = pd.read_parquet(f'{DATA_DIR}/metadata/items_metadata.parquet')

        train_users = train_interactions[USER].unique()
        train_items = train_interactions[ITEM].unique()

        item_emb = np.load(f'{DATA_DIR}/metadata/item_embeddings.npz')
        item_ids = item_emb[ITEM]
        item_embeddings = item_emb[EMBEDDING]

        mask = np.isin(item_ids, train_items)
        item_ids = item_ids[mask]
        item_embeddings = item_embeddings[mask]
        item_embeddings = item_embeddings[:, :self.content_embedding_size]

        users_metadata = users_metadata.merge(pd.DataFrame({USER: train_users}), on=USER, how='inner')
        items_metadata = items_metadata.merge(pd.DataFrame({ITEM: train_items}), on=ITEM, how='inner')
        items_metadata = items_metadata.merge(pd.DataFrame({ITEM: item_ids, EMBEDDING: item_embeddings.tolist()}), on=ITEM, how='inner')

        return (train_interactions, val_interactions), users_metadata, items_metadata

    def load_data(self):
        (train, val), users, items = self._load_data()
        tl, vl, ul, il = map(len, [train, val, users, items])

        for df in [train, val]:
            df[TARGET] = reduce(lambda x, y: x.add(y), [v * df[k] for k,v in INTERACTIONS_MAP.items()])
            df.drop(columns=INTERACTIONS_MAP.keys(), inplace=True)

        for df in [train, val, users, items]:
            df.drop(columns="train_interactions_rank", inplace=True, errors='ignore')

        agg_df = train.merge(items, on=ITEM)[[USER, ITEM, AUTHOR, TARGET]]
        users_agg, items_agg = [
            grouped.agg(
                positive=(TARGET, lambda x: (x>0).sum()),
                negative=(TARGET, lambda x: (x<0).sum()),
                count=(TARGET, 'count'),
                score=(TARGET, 'sum'),
            )
            for grouped in tqdm([agg_df.groupby(USER), agg_df.groupby(ITEM)])
        ]

        users_filtered = users_agg[
            (users_agg["count"] > 100)  # Не выгодно показывать клипы юзерам, которые не интересуются клипами
                & (users_agg["positive"].mean() > 0.05)  # Не выгодно показывать клипы юзерам которые смотрят, но не лайкают
                & (users_agg["negative"].mean() < 0.05)  # Не выгодно показывать клипы юзерам, которые много дизлайкают
        ]
        items_filtered = items_agg # Фильтровать айтемы и авторов не имеет смысла, мы предсказываем для всех данных айтемов

        users = users.merge(users_filtered[[]], left_on=USER, right_index=True, how='inner')
        print(f"Users: {ul:_} -> {len(users):_}")

        items = items.merge(items_filtered[[]], left_on=ITEM, right_index=True, how='inner')
        print(f"Items: {il:_} -> {len(items):_}")

        train = train\
            .merge(users_filtered[[]], left_on=USER, right_index=True, how='inner')\
            .merge(items_filtered[[]], left_on=ITEM, right_index=True, how='inner')
        print(f"Train: {tl:_} -> {len(train):_}")

        val = val \
            .merge(users_filtered[[]], left_on=USER, right_index=True, how='inner') \
            .merge(items_filtered[[]], left_on=ITEM, right_index=True, how='inner')
        print(f"Val: {vl:_} -> {len(val):_}")

        print(f"Train: {len(train):_} Val: {len(val):_} Users: {len(users):_} Items: {len(items):_}")
        return (train, val), users, items