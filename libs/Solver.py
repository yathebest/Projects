import heapq
from math import ceil
from typing import TypeVar, Hashable, Generic, Dict, List, Tuple, Iterable

import numpy as np
import polars as pl
from numpy.typing import NDArray
from ortools.graph.python.min_cost_flow import SimpleMinCostFlow
from tqdm.auto import tqdm

from .Trainer import Trainer
from .constants import *
from .utils import build_embeddings_map, count_polars

U = TypeVar('U', bound=Hashable)
I = TypeVar('I', bound=Hashable)

class Solver(Generic[I, U]):
    """
    Two-phase solver:
        1) Stream scores while 'collect_candidates' and keep top-L candidates per item.
        2) Solve reduced assignment with min-cost flow.

    :param trainer: model trainer with aggregated data
    :param predict_items: LazyFrame containing items to predict users for with columns [ITEM]
    :param candidates_to_keep: Number of candidates to keep per item (memory/quality knob).
    :param top_per_item: Number of users to select per item.
    :param max_per_user: Max times a user can be assigned across all items.
    :param score_scale: Scaling factor for converting float scores to integer costs for OR-Tools.
    """

    def __init__(self, trainer: Trainer,
                 predict_items: pl.LazyFrame,
                 candidates_to_keep: int,
                 top_per_item: int = 100, max_per_user: int = 100,
                 score_scale: float = 1e6):
        self.trainer = trainer
        self.predict_items = predict_items
        self.L = candidates_to_keep
        self.K = top_per_item
        self.R = max_per_user
        self.score_scale = score_scale

        # item -> dict(user -> best_score)
        self._item_to_user_score: Dict[I, Dict[U, float]] = {}
        # item -> heap of (score, user) used to evict smallest when > L
        self._item_heaps: Dict[I, List[Tuple[float, U]]] = {}

        self.predict_count = count_polars(self.predict_items)

    def collect_candidates(self, users_batch_size: int=1_000_000, items_batch_size: int=1_000_000):
        """
        :param users_batch_size: batching for users
        :param items_batch_size: batching for items
        """
        users_bar = tqdm(total=ceil(self.trainer.data_count / users_batch_size), position=0, desc="Users")
        score_items_bar = tqdm(total=ceil(self.predict_count), position=1, desc="Score Items")

        score_gen = self._score_generator(users_batch_size, items_batch_size)
        for batch_scores, users, items in score_gen:
            score_items_bar.reset()
            for col_idx, item in enumerate(items):
                mapping = self._item_to_user_score.get(item)
                if mapping is None:
                    mapping = {}
                    self._item_to_user_score[item] = mapping
                    self._item_heaps[item] = []

                heap = self._item_heaps[item]
                col_scores = batch_scores[:, col_idx]
                for row_idx, user in enumerate(users):
                    s = float(col_scores[row_idx])
                    prev = mapping.get(user)
                    if prev is None:
                        if len(mapping) < self.L:
                            mapping[user] = s
                            heapq.heappush(heap, (s, user))
                        else:
                            if heap and s > heap[0][0]:
                                mapping[user] = s
                                heapq.heappush(heap, (s, user))
                                self._evict_until_within_L(item)
                    else:
                        if s > prev:
                            mapping[user] = s
                            heapq.heappush(heap, (s, user))
                            if len(mapping) > self.L:
                                self._evict_until_within_L(item)
                score_items_bar.update(1)
            users_bar.update(1)

        for item in tqdm(list(self._item_to_user_score.keys()), desc="Clean up"):
            self._cleanup_item_heap(item)

    def _score_generator(self, users_batch_size: int, items_batch_size: int) -> Iterable[Tuple[NDArray, list[U], list[I]]]:
        items_bar = tqdm(total=ceil(self.predict_count / items_batch_size), position=2, desc="Predict Items")

        users_iterable = self.trainer.data.collect_batches(chunk_size=users_batch_size)
        for data_batch in users_iterable:
            users = data_batch[USER].to_list()

            user_embeddings = self.trainer.model.process_data_batch(data_batch, items_df=self.trainer.items_df, mode='predict')
            user_norms = np.linalg.norm(user_embeddings, axis=1, keepdims=True)
            user_norms[user_norms == 0] = 1e-12

            items_bar.reset()
            items, similarity_blocks = [], []
            iterable = self.predict_items.collect_batches(chunk_size=items_batch_size)
            for pred_batch in iterable:
                pred_items_batch = pred_batch[ITEM].to_list()
                items.extend(pred_items_batch)

                pred_item_to_emb = build_embeddings_map(self.trainer.items_df, pred_items_batch)

                item_embeddings = np.vstack([pred_item_to_emb[it] for it in pred_items_batch])  # (n_pred_batch, d)
                item_norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
                item_norms[item_norms == 0] = 1e-12

                similarity = (item_embeddings @ user_embeddings.T) / (item_norms * user_norms.T)  # (n_pred_batch, n_users_batch)
                similarity_blocks.append(similarity.T)

                items_bar.update(1)
            yield np.hstack(similarity_blocks), users, items

    def get_reduced_edges(self) -> List[Tuple[I, U, float]]:
        """
        Return list of (item, user, score) for current collected candidates.
        """
        edges = []
        for item, umap in self._item_to_user_score.items():
            for user, score in umap.items():
                edges.append((item, user, score))
        return edges

    def _evict_until_within_L(self, item: I):
        mapping = self._item_to_user_score[item]
        heap = self._item_heaps[item]
        # sanity: if heap empties but mapping still too large, rebuild a fresh heap from mapping
        while len(mapping) > self.L:
            if not heap:
                # rebuild heap from current mapping items
                heap = [(s, u) for u, s in mapping.items()]
                heapq.heapify(heap)
                self._item_heaps[item] = heap
                if not heap:
                    break  # nothing to pop
            s_pop, u_pop = heapq.heappop(heap)
            current = mapping.get(u_pop)
            if current is None:
                continue
            # use a tolerant comparison
            if abs(current - s_pop) < 1e-12:
                del mapping[u_pop]
            else:
                # stale entry, continue popping
                continue

    def _cleanup_item_heap(self, item: I):
        """Remove stale heap entries and ensure mapping and heap are consistent."""
        mapping = self._item_to_user_score[item]
        heap = self._item_heaps[item]
        new_heap = []
        for s, u in heap:
            # only keep entries that match mapping
            if u in mapping and abs(mapping[u] - s) < 1e-12:
                new_heap.append((s, u))
        heapq.heapify(new_heap)
        self._item_heaps[item] = new_heap
        # If mapping somehow exceeded L, evict down to L
        if len(mapping) > self.L:
            self._evict_until_within_L(item)

    def solve(self) -> Dict[I, List[Tuple[U, float]]]:
        """
        Solve assignment on the reduced candidate graph.

        :return: item_id -> list[K] of (user_id, score) sorted by score descending
        """
        # Build candidate list edges: (item, user, score)
        edges = []
        for item, user_map in self._item_to_user_score.items():
            for user, score in user_map.items():
                edges.append((item, user, score))

        if not edges:
            return {}

        # Unique node mappings
        items = sorted({e[0] for e in edges})
        users = sorted({e[1] for e in edges})
        item_to_idx = {it: idx for idx, it in enumerate(items)}
        user_to_idx = {u: idx for idx, u in enumerate(users)}

        total_item_demand = len(items) * self.K
        total_user_capacity = len(users) * self.R
        if total_user_capacity < total_item_demand:
            raise RuntimeError(
                f"Impossible to satisfy all demands: total user capacity {total_user_capacity} < total item demand {total_item_demand}."
                " Increase R or collect more candidate users per item (increase L)."
            )

        return self._solve_with_ortools(items, users, edges, item_to_idx, user_to_idx)

    def _solve_with_ortools(self, items, users, edges, item_to_idx, user_to_idx):
        # Build min-cost flow graph with nodes:
        # source (0), item nodes 1..I, user nodes I+1..I+U, sink = I+U+1
        I = len(items)
        U = len(users)
        source = 0
        item_base = 1
        user_base = item_base + I
        sink = user_base + U

        min_cost_flow = SimpleMinCostFlow()

        # helper to add arc and record arc ids for later flow extraction
        arc_identifiers = []  # ((item_idx, user_idx), arc_index_in_or_tools)
        max_score = max(s for (_, _, s) in edges)
        # Add source -> item arcs (capacity K, cost 0)
        for i_idx in range(I):
            min_cost_flow.add_arc_with_capacity_and_unit_cost(source, item_base + i_idx, self.K, 0)

        # Add user -> sink arcs (capacity R, cost 0)
        for u_idx in range(U):
            min_cost_flow.add_arc_with_capacity_and_unit_cost(user_base + u_idx, sink, self.R, 0)

        # Add item->user arcs for every candidate (capacity 1, cost based on score)
        # To maximize score, set cost = int(round((max_score - score) * scale))
        for (item, user, score) in edges:
            i_idx = item_to_idx[item]
            u_idx = user_to_idx[user]
            # transform score -> non-negative integer cost (smaller cost = better score)
            # scale and round carefully. Ensure cost is int.
            cost = int(round((max_score - score) * self.score_scale))
            arc_id = min_cost_flow.add_arc_with_capacity_and_unit_cost(item_base + i_idx, user_base + u_idx, 1, cost)
            arc_identifiers.append(((i_idx, u_idx), arc_id))

        # Set supplies: source has +total demand, sink has -total demand
        total_demand = len(items) * self.K
        min_cost_flow.set_node_supply(source, total_demand)
        min_cost_flow.set_node_supply(sink, -total_demand)

        status = min_cost_flow.solve_max_flow_with_min_cost()
        if status != min_cost_flow.OPTIMAL:
            raise NotImplementedError()

        # Extract chosen edges (flow == 1 on item->user arcs)
        # Build reverse maps
        idx_to_item = {v: k for k, v in item_to_idx.items()}
        idx_to_user = {v: k for k, v in user_to_idx.items()}
        assignments = {item: [] for item in items}
        # For each arc, query flow
        for ((i_idx, u_idx), arc_id) in arc_identifiers:
            if min_cost_flow.flow(arc_id) > 0:
                item = idx_to_item[i_idx]
                user = idx_to_user[u_idx]
                score = self._item_to_user_score[item][user]
                assignments[item].append((user, score))

        # Ensure each list sorted by descending score and limited to K
        for it in assignments:
            assignments[it] = sorted(assignments[it], key=lambda x: -x[1])[:self.K]
        return assignments
