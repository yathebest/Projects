import heapq
from math import ceil
from typing import TypeVar, Hashable, Generic, Dict, List, Tuple, Iterable

import numpy as np
import polars as pl
from numpy.typing import NDArray
from ortools.graph.python.min_cost_flow import SimpleMinCostFlow
from tqdm.auto import tqdm

from .constants import *
from .utils import count_polars

U = TypeVar('U', bound=Hashable)
I = TypeVar('I', bound=Hashable)

class Solver(Generic[I, U]):
    """
    Two-phase solver:
      1) Stream scores from `score_generator` and keep top-L candidates per item.
      2) Solve reduced assignment with min-cost flow (OR-Tools) or greedy fallback.

    Parameters
    ----------
    candidates_to_keep : int
        Number of candidates to keep per item (memory/quality knob).
    top_per_item : int
        Number of users to select per item.
    max_per_user : int
        Max times a user can be assigned across all items.
    use_ortools : bool
        If True, prefer OR-Tools solver when available; otherwise use greedy fallback.
    score_scale : float
        Scaling factor for converting float scores to integer costs for OR-Tools.
    """

    def __init__(self, candidates_to_keep: int, top_per_item: int, max_per_user: int,
                 use_ortools: bool = True, score_scale: float = 1e6):
        self.L = candidates_to_keep
        self.K = top_per_item
        self.R = max_per_user
        self.use_ortools = use_ortools
        self.score_scale = score_scale

        # item -> dict(user -> best_score)
        self._item_to_user_score: Dict[I, Dict[U, float]] = {}
        # item -> heap of (score, user) used to evict smallest when > L
        self._item_heaps: Dict[I, List[Tuple[float, U]]] = {}

    def collect_candidates(self, data: pl.LazyFrame, val: pl.LazyFrame, items_df: pl.LazyFrame, model,
                           users_batch_size: int=1_000_000, items_batch_size: int=1_000_000):
        """
        :param data: LazyFrame with columns [ITEM, USER, TIME_INDEX, TARGET], where USER, TIME_INDEX, TARGET are List[n_user_interactions]
        :param val: LazyFrame with columns [ITEM] - data to predict
        :param items_df: LazyFrame with index [ITEM] and columns [EMBEDDING] - contains all items with their embeddings
        :param users_batch_size: batching for users
        :param items_batch_size: batching for items
        """
        score_gen = Solver._score_generator(data, val, items_df, model, users_batch_size, items_batch_size)
        for batch_scores, users, items in score_gen:
            # iterate columns (items) to avoid iterating over full matrix
            for col_idx, item in enumerate(items):
                # lazy init per item
                mapping = self._item_to_user_score.get(item)
                if mapping is None:
                    mapping = {}
                    self._item_to_user_score[item] = mapping
                    self._item_heaps[item] = []

                heap = self._item_heaps[item]
                # extract column vector of scores
                # assume batch_scores is numpy-like with indexing [row, col]
                col_scores = batch_scores[:, col_idx]
                for row_idx, user in enumerate(users):
                    s = float(col_scores[row_idx])
                    # if user already present, possibly update
                    prev = mapping.get(user)
                    if prev is None:
                        # add if there's room
                        if len(mapping) < self.L:
                            mapping[user] = s
                            heapq.heappush(heap, (s, user))
                        else:
                            # If new score beats current min, add and evict
                            # heap[0] is smallest (min-heap)
                            if heap and s > heap[0][0]:
                                mapping[user] = s
                                heapq.heappush(heap, (s, user))
                                # Evict until mapping size <= L and heap top matches mapping
                                self._evict_until_within_L(item)
                    else:
                        # user already present: update if this score is larger
                        if s > prev:
                            mapping[user] = s
                            heapq.heappush(heap, (s, user))
                            # If mapping grew beyond L (shouldn't unless we inserted new user), evict
                            if len(mapping) > self.L:
                                self._evict_until_within_L(item)

        # After all batches, optionally prune heap leftovers (make mapping consistent)
        for item in list(self._item_to_user_score.keys()):
            self._cleanup_item_heap(item)

    @staticmethod
    def _score_generator(data: pl.LazyFrame, val: pl.LazyFrame, items_df: pl.LazyFrame, model,
                         users_batch_size: int=1_000_000, items_batch_size: int=1_000_000
                         ) -> Iterable[Tuple[NDArray, list[U], list[I]]]:
        n_data = count_polars(data)
        n_val = count_polars(val)

        data_bar = tqdm(total=ceil(n_data / users_batch_size), position=0, desc="data")
        val_bar = tqdm(total=ceil(n_val / items_batch_size), position=1, desc="val")

        data_bar.reset()
        for data_batch in data.collect_batches(chunk_size=users_batch_size):
            users = data_batch[USER].to_list()

            user_embeddings = model.process_data_batch(data_batch, items_df=items_df, items_batch_size=items_batch_size)
            user_norms = np.linalg.norm(user_embeddings, axis=1, keepdims=True)
            user_norms[user_norms == 0] = 1e-12

            val_bar.reset()
            items, similarity_blocks = [], []
            for val_batch in val.collect_batches(chunk_size=items_batch_size):
                val_items_batch = val_batch[ITEM].to_list()
                items.extend(val_items_batch)

                val_items_df_batch = items_df.filter(pl.col(ITEM).is_in(val_items_batch)).select([ITEM, EMBEDDING]).collect()
                val_item_to_emb = {r[0]: np.asarray(r[1], dtype=np.float64) for r in val_items_df_batch.rows()}

                item_embeddings = np.vstack([val_item_to_emb[it] for it in val_items_batch])  # (n_val_batch, d)
                item_norms = np.linalg.norm(item_embeddings, axis=1, keepdims=True)
                item_norms[item_norms == 0] = 1e-12

                similarity = (item_embeddings @ user_embeddings.T) / (item_norms * user_norms.T)  # (n_val_batch, n_users_batch)
                similarity_blocks.append(similarity.T)

                val_bar.update(1)
            data_bar.update(1)
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

        Returns
        -------
        assignments : dict
          item_id -> list of (user_id, score), length K for each item (if feasible)
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

        # If OR-Tools is available and requested, use min-cost flow
        if self.use_ortools:
            return self._solve_with_ortools(items, users, edges, item_to_idx, user_to_idx)
        else:
            raise NotImplementedError()

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