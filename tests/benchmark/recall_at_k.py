# Path: tests/benchmark/recall_at_k.py
# Recall@K evaluation for multimodal retrieval.

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from vlm_engineering.retrieval.in_memory import SearchResult
from vlm_engineering.vectorstore.protocol import VectorStore


def recall_at_k(
    store: VectorStore,
    query_embedding: np.ndarray,
    relevant_ids: Sequence[str],
    *,
    top_k: int = 10,
) -> float:
    """Compute Recall@K for a single query.

    Recall@K measures the proportion of queries for which at least one
    relevant item appears in the top-K results. For a single query, the
    score is 1 if a relevant item is found in the top-K, and 0 otherwise.

    Args:
        store:
            An implemented :class:`VectorStore` with indexed items.
        query_embedding:
            1-D embedding vector for the query.
        relevant_ids:
            Logical chunk IDs that are considered relevant for this query.
        top_k:
            Number of top results to consider.

    Returns:
        Recall@K score: 1.0 if at least one relevant item is in the top-K,
        otherwise 0.0.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    results: list[SearchResult] = store.search(
        query_embedding, top_k=max(top_k, store.count())
    )
    retrieved_ids = {str(result.item) if result.item is not None else "" for result in results}
    # Check if any of the relevant IDs appear in the retrieved set.
    has_relevant = bool(set(relevant_ids) & set(retrieved_ids))
    return 1.0 if has_relevant else 0.0


def average_recall_at_k(
    store: VectorStore,
    queries: Sequence[dict[str, Any]],
    *,
    top_k: int = 10,
) -> float:
    """Compute average Recall@K across multiple queries.

    Each query dict should have:
        - "embedding": 1-D query embedding vector
        - "relevant_ids": sequence of logical chunk IDs relevant to the query

    Args:
        store:
            An implemented :class:`VectorStore` with indexed items.
        queries:
            Sequence of query dicts with ``embedding`` and ``relevant_ids``.
        top_k:
            Number of top results to consider per query.

    Returns:
        Mean Recall@K across all queries (value in [0, 1]).
    """
    if not queries:
        return 0.0
    scores = [
        recall_at_k(store, q["embedding"], q["relevant_ids"], top_k=top_k)
        for q in queries
    ]
    return float(np.mean(scores))