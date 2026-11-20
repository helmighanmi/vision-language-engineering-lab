# Path: tests/benchmark/ndcg_at_k.py
# nDCG@K evaluation for multimodal retrieval.

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from vlm_engineering.retrieval.in_memory import SearchResult
from vlm_engineering.vectorstore.protocol import VectorStore


def _dcg(scores: Sequence[float]) -> float:
    """Compute Discounted Cumulative Gain at rank positions."""
    if not scores:
        return 0.0
    gains = [2.0 ** s - 1.0 for s in scores]
    discounts = np.log2(np.arange(2, len(gains) + 2, dtype=np.float64))
    return float(np.sum(gains / discounts))


def _ideal_dcg(relevance: Sequence[int], k: int) -> float:
    """Compute the ideal DCG (IDCG) for a given relevance pattern.

    relevance: binary relevance labels (1 = relevant, 0 = not relevant),
    sorted by relevance descending (ideal ranking).
    """
    # Sort relevance descending so the most relevant items come first.
    rel_sorted = sorted(relevance, reverse=True) if isinstance(relevance, list) else sorted(relevance, reverse=True)
    # Take top-k
    top = rel_sorted[:k]
    return _dcg([1.0 if r else 0.0 for r in top])


def ndcg_at_k(
    store: VectorStore,
    query_embedding: np.ndarray,
    relevant_ids: Sequence[str],
    *,
    top_k: int = 10,
) -> float:
    """Compute nDCG@K for a single query.

    nDCG@K normalizes DCG@K by the ideal DCG (IDCG@K), so the score is
    always in [0, 1], where 1.0 is a perfect ranking.

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
        nDCG@K score in [0, 1]; returns 0.0 if no results are returned
        or IDCG is zero.
    """
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    results: list[SearchResult] = store.search(
        query_embedding, top_k=max(top_k, store.count())
    )
    retrieved_ids = [str(result.item) if result.item is not None else "" for result in results]
    # Build relevance labels: 1 if the item is relevant, 0 otherwise.
    rel = [1 if rid in retrieved_ids else 0 for rid in relevant_ids]
    # Compute DCG@K at the actual rank order.
    dcg = _dcg([float(r.score) for r in results[:top_k]])
    # Compute IDCG@K (ideal ranking gives highest relevance to the top positions).
    idcg = _ideal_dcg(rel, top_k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def average_ndcg_at_k(
    store: VectorStore,
    queries: Sequence[dict[str, Any]],
    *,
    top_k: int = 10,
) -> float:
    """Compute average nDCG@K across multiple queries.

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
        Mean nDCG@K across all queries (value in [0, 1]).
    """
    if not queries:
        return 0.0
    scores = [
        ndcg_at_k(store, q["embedding"], q["relevant_ids"], top_k=top_k)
        for q in queries
    ]
    return float(np.mean(scores))