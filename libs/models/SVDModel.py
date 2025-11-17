import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from tqdm.auto import trange

from ..constants import *


class SVDModel:
    def __init__(self, n_factors=10):
        self.n_factors = n_factors
        self.user_id_map = None
        self.id_user_map = None
        self.item_id_map = None
        self.id_item_map = None
        self.user_factors = None
        self.item_factors = None
        self.sparse_interactions = None
        self.sparse_interactions_csr = None
        self.sparse_interactions_csc = None

    def fit(self, interactions_df):
        self._create_mappings(interactions_df)
        self._create_sparse_matrix(interactions_df)
        self._factorize()
        return self

    def recommend_items_for_user(self, user_id, n_recommendations=10, exclude_interacted=True):
        if user_id not in self.user_id_map:
            return [], []

        user_idx = self.user_id_map[user_id]
        user_vector = self.user_factors[user_idx]
        scores = self.item_factors.dot(user_vector)

        if exclude_interacted:
            interacted_mask = self._get_interacted_items(user_idx)
            scores[interacted_mask] = -np.inf

        top_item_indices = np.argpartition(scores, -n_recommendations)[-n_recommendations:]
        top_item_indices = top_item_indices[np.argsort(scores[top_item_indices])[::-1]]
        top_scores = scores[top_item_indices]

        return [self.id_item_map[idx] for idx in top_item_indices], top_scores

    def recommend_users_for_item(self, item_id, n_recommendations=10, exclude_interacted=True):
        if item_id not in self.item_id_map:
            return [], []

        item_idx = self.item_id_map[item_id]
        item_vector = self.item_factors[item_idx]
        scores = self.user_factors.dot(item_vector)

        if exclude_interacted:
            interacted_mask = self._get_interacted_users(item_idx)
            scores[interacted_mask] = -np.inf

        top_user_indices = np.argpartition(scores, -n_recommendations)[-n_recommendations:]
        top_user_indices = top_user_indices[np.argsort(scores[top_user_indices])[::-1]]
        top_scores = scores[top_user_indices]

        return [self.id_user_map[idx] for idx in top_user_indices], top_scores

    def batch_recommend_items_for_users(self, user_ids, n_recommendations=10, exclude_interacted=True, batch_size=1000):
        recommendations = {}
        valid_users = [uid for uid in user_ids if uid in self.user_id_map]

        if not valid_users:
            return {}

        valid_indices = np.array([self.user_id_map[uid] for uid in valid_users])

        for i in trange(0, len(valid_indices), batch_size, desc='Predict batches'):
            batch_indices = valid_indices[i:i+batch_size]
            batch_users = valid_users[i:i+batch_size]

            user_vectors = self.user_factors[batch_indices]
            batch_scores = user_vectors @ self.item_factors.T

            if exclude_interacted:
                for j, user_idx in enumerate(batch_indices):
                    interacted_mask = self._get_interacted_items(user_idx)
                    batch_scores[j, interacted_mask] = -np.inf

            for j, user_id in enumerate(batch_users):
                scores = batch_scores[j]
                top_indices = np.argpartition(scores, -n_recommendations)[-n_recommendations:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

                recommendations[user_id] = (
                    [self.id_item_map[idx] for idx in top_indices],
                    scores[top_indices]
                )

        return recommendations

    def batch_recommend_users_for_items(self, item_ids, n_recommendations=10, exclude_interacted=True, batch_size=1000):
        recommendations = {}
        valid_items = [iid for iid in item_ids if iid in self.item_id_map]

        if not valid_items:
            return {}

        valid_indices = np.array([self.item_id_map[iid] for iid in valid_items])

        for i in trange(0, len(valid_indices), batch_size, desc='Predict batches'):
            batch_indices = valid_indices[i:i+batch_size]
            batch_items = valid_items[i:i+batch_size]

            batch_scores = self.user_factors @ self.item_factors[batch_indices].T

            if exclude_interacted:
                for j, item_idx in enumerate(batch_indices):
                    interacted_mask = self._get_interacted_users(item_idx)
                    batch_scores[interacted_mask, j] = -np.inf

            for j, item_id in enumerate(batch_items):
                scores = batch_scores[:, j]
                top_indices = np.argpartition(scores, -n_recommendations)[-n_recommendations:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

                recommendations[item_id] = (
                    [self.id_user_map[idx] for idx in top_indices],
                    scores[top_indices]
                )

        return recommendations

    def predict_all_scores(self):
        return self.user_factors @ self.item_factors.T

    def predict_score(self, user_id, item_id):
        if user_id not in self.user_id_map or item_id not in self.item_id_map:
            return 0.0

        user_idx = self.user_id_map[user_id]
        item_idx = self.item_id_map[item_id]
        return np.dot(self.user_factors[user_idx], self.item_factors[item_idx])

    def get_similar_items(self, item_id, n_similar=10):
        if item_id not in self.item_id_map:
            return [], []

        item_idx = self.item_id_map[item_id]
        target_vector = self.item_factors[item_idx]

        norms = np.linalg.norm(self.item_factors, axis=1)
        target_norm = np.linalg.norm(target_vector)
        similarities = self.item_factors.dot(target_vector) / (norms * target_norm)
        similarities[item_idx] = -1

        similar_indices = np.argsort(similarities)[::-1][:n_similar]
        similar_scores = similarities[similar_indices]

        return [self.id_item_map[idx] for idx in similar_indices], similar_scores

    def _create_mappings(self, interactions_df):
        unique_users = sorted(interactions_df[USER].unique().tolist())
        unique_items = sorted(interactions_df[ITEM].unique().tolist())

        self.user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.id_user_map = {idx: user_id for user_id, idx in self.user_id_map.items()}

        self.item_id_map = {item_id: idx for idx, item_id in enumerate(unique_items)}
        self.id_item_map = {idx: item_id for item_id, idx in self.item_id_map.items()}

    def _create_sparse_matrix(self, interactions_df):
        user_indices = interactions_df[USER].map(self.user_id_map)
        item_indices = interactions_df[ITEM].map(self.item_id_map)
        targets = interactions_df[TARGET] if TARGET in interactions_df else np.ones(len(interactions_df))

        self._interaction_user_indices = user_indices
        self._interaction_item_indices = item_indices

        self.sparse_interactions = coo_matrix(
            (targets, (user_indices, item_indices)),
            shape=(len(self.user_id_map), len(self.item_id_map))
        )
        self.sparse_interactions_csr = self.sparse_interactions.tocsr()
        self.sparse_interactions_csc = self.sparse_interactions.tocsc()

    def _factorize(self):
        k = min(self.n_factors, min(self.sparse_interactions.shape) - 1)
        U, sigma, Vt = svds(self.sparse_interactions_csr, k=k)
        self.user_factors = U * sigma
        self.item_factors = Vt.T

    def _get_interacted_items(self, user_idx):
        if user_idx >= self.sparse_interactions_csr.shape[0]:
            return np.zeros(len(self.item_id_map), dtype=bool)

        indptr = self.sparse_interactions_csr.indptr
        indices = self.sparse_interactions_csr.indices

        start, end = indptr[user_idx], indptr[user_idx + 1]
        interacted_indices = indices[start:end]

        mask = np.zeros(len(self.item_id_map), dtype=bool)
        mask[interacted_indices] = True
        return mask

    def _get_interacted_users(self, item_idx):
        if item_idx >= self.sparse_interactions_csc.shape[1]:
            return np.zeros(len(self.user_id_map), dtype=bool)

        indptr = self.sparse_interactions_csc.indptr
        indices = self.sparse_interactions_csc.indices

        start, end = indptr[item_idx], indptr[item_idx + 1]
        interacted_indices = indices[start:end]

        mask = np.zeros(len(self.user_id_map), dtype=bool)
        mask[interacted_indices] = True
        return mask