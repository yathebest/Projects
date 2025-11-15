import os
from typing import Tuple, Optional

import numpy as np
import polars as pl
from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download

from .constants import *
from .utils import count_polars


class Loader:
    MAX_WEEKS = 25
    TEST_WEEK = 25
    """
    :param subsample_name: name of subsample to download
    :param content_embedding_size: items embeddings dimension
    :param all_weeks: num of weeks to load for train and val
    :param val_weeks: num of weeks to keep for val, should be [0...all_weeks]
    :param batch_size: chuck size to aggregate data, if not provided data will be aggregated in 1 chunk
    """
    def __init__(self, subsample_name: str, content_embedding_size: int = 64,
                 all_weeks: int = MAX_WEEKS, val_weeks: int = 0,
                 batch_size: Optional[int] = None):
        self.subsample_name = subsample_name
        self.content_embedding_size = content_embedding_size
        self.batch_size = batch_size
        self.all_weeks = all_weeks
        self.val_weeks = val_weeks

        if not (0 <= self.val_weeks <= self.all_weeks <= Loader.MAX_WEEKS):
            raise NotImplementedError()

    def _get_files(self) -> Tuple[Tuple[list, list], list, list]:
        if self.val_weeks > 0:
            val_start = self.all_weeks - self.val_weeks
            train_weeks = list(range(val_start))
            val_weeks = list(range(val_start, self.all_weeks))
        else:
            train_weeks = list(range(self.all_weeks))
            val_weeks = []

        train_interactions_files = [
            f'subsamples/{self.subsample_name}/train/week_{i:02}.parquet' for i in train_weeks
        ]
        val_interactions_files = [
            f'subsamples/{self.subsample_name}/train/week_{i:02}.parquet' for i in val_weeks
        ]
        test_files = [
            f'subsamples/{self.subsample_name}/validation/week_{Loader.TEST_WEEK:02}.parquet'
        ]

        metadata_files = ['metadata/users_metadata.parquet', 'metadata/items_metadata.parquet', 'metadata/item_embeddings.npz']
        return (train_interactions_files, val_interactions_files), test_files, metadata_files

    def _download_files(self, files: list = None):
        if files is None:
            (t, v), test_files, m = self._get_files()
            files = t + v + test_files + m

        for file in tqdm(files):
            hf_hub_download(
                repo_id='deepvk/VK-LSVD', repo_type='dataset',
                filename=file, local_dir=DATA_DIR
            )

    def _ensure_files_exist(self, files: Optional[list[str]] = None):
        if files is None:
            (train_files, val_files), test_files, metadata_files = self._get_files()
            all_files = train_files + val_files + test_files + metadata_files
        else:
            all_files = files

        missing_files = [f for f in all_files if not os.path.exists(f'{DATA_DIR}/{f}')]

        if missing_files:
            print(f"{len(missing_files)} files not found, downloading...")
            self._download_files(missing_files)

    def _get_num_batches(self, data: pl.LazyFrame) -> int:
        if self.batch_size is not None:
            return (count_polars(data) / self.batch_size).__ceil__()
        return 1

    def _create_target_expression(self) -> pl.Expr:
        target_expr = pl.lit(0)
        for interaction, weight in INTERACTIONS_MAP.items():
            target_expr = target_expr + pl.col(interaction) * weight
        return target_expr.alias(TARGET)

    def _load_metadata(self) -> Tuple[pl.LazyFrame, pl.LazyFrame, np.ndarray, np.ndarray]:
        users_metadata = pl.scan_parquet(f'{DATA_DIR}/metadata/users_metadata.parquet')
        items_metadata = pl.scan_parquet(f'{DATA_DIR}/metadata/items_metadata.parquet')

        item_emb = np.load(f'{DATA_DIR}/metadata/item_embeddings.npz')
        item_ids = item_emb[ITEM]
        item_embeddings = item_emb[EMBEDDING]

        return users_metadata, items_metadata, item_ids, item_embeddings

    def _create_lazy_datasets(self) -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
        # train/val lists come from _get_files; test_files are returned and used here
        (train_files, val_files), test_files, _ = self._get_files()
        train, val, test = [], [], []
        time_offset = 0

        for file in train_files:  # first train by time
            df = pl.scan_parquet(f'{DATA_DIR}/{file}')
            df = df.with_columns((pl.row_index()+pl.lit(time_offset)).alias(TIME_INDEX))
            time_offset += count_polars(df)
            train.append(df)

        for file in val_files:  # then val by time (these are taken from 0..24 depending on val_weeks)
            df = pl.scan_parquet(f'{DATA_DIR}/{file}')
            df = df.with_columns((pl.row_index()+pl.lit(time_offset)).alias(TIME_INDEX))
            time_offset += count_polars(df)
            val.append(df)

        for file in test_files:  # finally test (week_25)
            df = pl.scan_parquet(f'{DATA_DIR}/{file}')
            df = df.with_columns((pl.row_index()+pl.lit(time_offset)).alias(TIME_INDEX))
            time_offset += count_polars(df)
            test.append(df)

        return (pl.concat(train) if train else pl.LazyFrame(pl.DataFrame()),
                pl.concat(val) if val else pl.LazyFrame(pl.DataFrame()),
                pl.concat(test) if test else pl.LazyFrame(pl.DataFrame()))

    def _get_unique_items_users(self, data_lazy: pl.LazyFrame) -> Tuple[pl.Series, pl.Series]:
        items_set = set()
        users_set = set()

        iterable: tqdm_asyncio[pl.DataFrame] = tqdm(
            data_lazy.collect_batches(chunk_size=self.batch_size),
            total=self._get_num_batches(data_lazy),
        ) if self.batch_size is not None else [data_lazy.collect()]

        for batch in iterable:
            if ITEM in batch.columns:
                items_set.update(batch[ITEM].unique().to_list())
            if USER in batch.columns:
                users_set.update(batch[USER].unique().to_list())

        items_series = pl.Series(ITEM, list(items_set))
        users_series = pl.Series(USER, list(users_set))
        return items_series, users_series

    def _filter_embeddings(self, item_ids: np.ndarray, item_embeddings: np.ndarray, items: pl.Series):
        mask = np.isin(item_ids, items.to_list())
        filtered_ids = item_ids[mask]
        filtered_embeddings = item_embeddings[mask]
        filtered_embeddings = filtered_embeddings[:, :self.content_embedding_size]

        return filtered_ids, filtered_embeddings

    def _process_metadata(self, users_metadata: pl.LazyFrame, items_metadata: pl.LazyFrame,
                          users: pl.Series, items: pl.Series,
                          filtered_ids: np.ndarray, filtered_embeddings: np.ndarray):
        users_df = pl.LazyFrame({USER: users})
        items_df = pl.LazyFrame({ITEM: items})

        users_metadata = users_metadata.join(users_df, on=USER, how='inner')
        items_metadata = items_metadata.join(items_df, on=ITEM, how='inner')

        embeddings_df = pl.LazyFrame({ITEM: filtered_ids, EMBEDDING: filtered_embeddings})
        items_metadata = items_metadata.join(embeddings_df, on=ITEM, how='inner')

        return users_metadata, items_metadata

    def _compute_aggregates(self, data_lazy: pl.LazyFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        users_acc = None
        items_acc = None

        iterable: tqdm_asyncio[pl.DataFrame] = tqdm(
            data_lazy.collect_batches(chunk_size=self.batch_size),
            total=self._get_num_batches(data_lazy),
        ) if self.batch_size is not None else [data_lazy.collect()]

        for batch in iterable:
            batch = batch.with_columns([self._create_target_expression()])

            user_batch = batch.group_by(USER).agg([
                pl.col(TARGET).filter(pl.col(TARGET) > 0).count().alias("positive"),
                pl.col(TARGET).filter(pl.col(TARGET) < 0).count().alias("negative"),
                pl.col(TARGET).count().alias("count"),
                pl.col(TARGET).sum().alias("score")
            ])

            item_batch = batch.group_by(ITEM).agg([
                pl.col(TARGET).filter(pl.col(TARGET) > 0).count().alias("positive"),
                pl.col(TARGET).filter(pl.col(TARGET) < 0).count().alias("negative"),
                pl.col(TARGET).count().alias("count"),
                pl.col(TARGET).sum().alias("score")
            ])

            if users_acc is None:
                users_acc = user_batch
            else:
                users_acc = pl.concat([users_acc, user_batch]).group_by(USER).agg([
                    pl.col("positive").sum().alias("positive"),
                    pl.col("negative").sum().alias("negative"),
                    pl.col("count").sum().alias("count"),
                    pl.col("score").sum().alias("score")
                ])

            if items_acc is None:
                items_acc = item_batch
            else:
                items_acc = pl.concat([items_acc, item_batch]).group_by(ITEM).agg([
                    pl.col("positive").sum().alias("positive"),
                    pl.col("negative").sum().alias("negative"),
                    pl.col("count").sum().alias("count"),
                    pl.col("score").sum().alias("score")
                ])

        if users_acc is None:
            users_acc = pl.DataFrame({USER: [], "positive": [], "negative": [], "count": [], "score": []})
        if items_acc is None:
            items_acc = pl.DataFrame({ITEM: [], "positive": [], "negative": [], "count": [], "score": []})

        return users_acc, items_acc

    def _filter_data(self, users_agg: pl.DataFrame, items_agg: pl.DataFrame,
                     users_metadata: pl.LazyFrame, items_metadata: pl.LazyFrame,
                     train: pl.LazyFrame, val: pl.LazyFrame, test: pl.LazyFrame):
        users_filtered = users_agg.filter(
            (pl.col("count") > 100) &
            (pl.col("positive") / pl.col("count") > 0.05) &
            (pl.col("negative") / pl.col("count") < 0.05)
        )
        items_filtered = items_agg  # keep as-is (you can add item-level filters here)

        # convert filtered user/item lists to Python lists (should fit memory)
        users_keep = users_filtered.select(USER).to_series().to_list()
        items_keep = items_filtered.select(ITEM).to_series().to_list()

        users_metadata = users_metadata.join(
            pl.DataFrame({USER: users_keep}).lazy(), on=USER, how='inner'
        )
        items_metadata = items_metadata.join(
            pl.DataFrame({ITEM: items_keep}).lazy(), on=ITEM, how='inner'
        )

        # Build lazy selections for joins on large datasets
        users_keep_lazy = pl.DataFrame({USER: users_keep}).lazy().select(USER)
        items_keep_lazy = pl.DataFrame({ITEM: items_keep}).lazy().select(ITEM)

        train_filtered = (train
                          .join(users_keep_lazy, on=USER, how='inner')
                          .join(items_keep_lazy, on=ITEM, how='inner'))

        val_filtered = (val
                        .join(users_keep_lazy, on=USER, how='inner')
                        .join(items_keep_lazy, on=ITEM, how='inner'))

        test_filtered = (test
                         .join(users_keep_lazy, on=USER, how='inner')
                         .join(items_keep_lazy, on=ITEM, how='inner'))

        return train_filtered, val_filtered, test_filtered, users_metadata, items_metadata

    def load_data(self, convert_to_pandas=True, filter_data=True):
        self._ensure_files_exist()

        print("Load metadata")
        users_metadata, items_metadata, item_ids, item_embeddings = self._load_metadata()

        print("Create lazy interaction datasets")
        train_lazy, val_lazy, test_lazy = self._create_lazy_datasets()

        print("Get unique users/items")
        train_items, train_users = self._get_unique_items_users(train_lazy)
        val_items, val_users = self._get_unique_items_users(val_lazy)
        test_items, test_users = self._get_unique_items_users(test_lazy)
        all_items = pl.concat([train_items, val_items, test_items]).unique()
        all_users = pl.concat([train_users, val_users, test_users]).unique()

        print("Filter embeddings")
        filtered_ids, filtered_embeddings = self._filter_embeddings(
            item_ids, item_embeddings, all_items
        )

        print("Process metadata")
        users_metadata, items_metadata = self._process_metadata(
            users_metadata, items_metadata, all_users, all_items,
            filtered_ids, filtered_embeddings
        )

        if not filter_data:
            if convert_to_pandas:
                return  (((train_lazy.collect().to_pandas(), val_lazy.collect().to_pandas()), test_lazy.collect().to_pandas()),
                         users_metadata.collect().to_pandas(), items_metadata.collect().to_pandas())

            return (train_lazy, val_lazy), test_lazy, users_metadata, items_metadata

        print("Compute aggregates")
        users_agg, items_agg = self._compute_aggregates(train_lazy)

        print("Filter interactions")
        train_filtered, val_filtered, test_filtered, users_metadata, items_metadata = self._filter_data(
            users_agg, items_agg, users_metadata, items_metadata, train_lazy, val_lazy, test_lazy
        )

        print("Finalize interactions")
        target_expr = self._create_target_expression()

        train_final = (train_filtered
                       .with_columns([target_expr])
                       .drop(list(INTERACTIONS_MAP.keys()), strict=False)
                       .drop("train_interactions_rank", strict=False)
                       .filter(pl.col(TARGET).ne(0)))

        val_final = (val_filtered
                     .with_columns([target_expr])
                     .drop(list(INTERACTIONS_MAP.keys()), strict=False)
                     .drop("train_interactions_rank", strict=False)
                     .filter(pl.col(TARGET).ge(0)))

        test_final = (test_filtered
                      .with_columns([target_expr])
                      .drop(list(INTERACTIONS_MAP.keys()), strict=False)
                      .drop("train_interactions_rank", strict=False)
                      .filter(pl.col(TARGET).ge(0)))

        users_metadata = users_metadata.drop("train_interactions_rank", strict=False)
        items_metadata = items_metadata.drop("train_interactions_rank", strict=False)

        print("Count")
        print(f"Train: {count_polars(train_final):_} Val: {count_polars(val_final):_} Test: {count_polars(test_final):_} "
              f"Users: {count_polars(users_metadata):_} Items: {count_polars(items_metadata):_}")

        if convert_to_pandas:
            print("Convert to pandas")
            return  (((train_final.collect().to_pandas(), val_final.collect().to_pandas()), test_final.collect().to_pandas()),
                     users_metadata.collect().to_pandas(), items_metadata.collect().to_pandas())

        return ((train_final, val_final), test_final), users_metadata, items_metadata
