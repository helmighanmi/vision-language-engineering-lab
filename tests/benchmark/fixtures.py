# Path: tests/benchmark/fixtures.py
# Deterministic test fixtures for Recall@K and nDCG@K benchmarks.

from __future__ import annotations

from typing import Any

import numpy as np

from vlm_engineering.vectorstore import MemoryVectorStore


def make_deterministic_embeddings(
    num_vectors: int, dimension: int = 768, seed: int = 42
) -> np.ndarray:
    """Generate deterministic unit-length embeddings for testing."""
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(num_vectors, dimension))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return (vectors / norms).astype(np.float32)


def make_test_store(
    num_items: int = 10,
    embedding_dimension: int = 768,
    seed: int = 42,
) -> tuple[MemoryVectorStore, list[str], dict[int, list[str]], list[np.ndarray]]:
    """Create a deterministic vector store with known relevance relationships."""
    store = MemoryVectorStore()
    vectors = make_deterministic_embeddings(num_items, embedding_dimension, seed)
    all_item_ids = [f"item-{i}" for i in range(num_items)]
    store.add(vectors, all_item_ids)

    query_relevant_ids: dict[int, list[str]] = {
        0: ["item-0", "item-1", "item-2"],
        1: ["item-3", "item-4"],
        2: ["item-5", "item-6", "item-7", "item-8", "item-9"],
    }
    query_embeddings = [
        make_deterministic_embeddings(1, embedding_dimension, seed + 100 + q_idx)[0]
        for q_idx in range(3)
    ]
    return store, all_item_ids, query_relevant_ids, query_embeddings


def make_recall_benchmark_fixture(
    num_items: int = 10,
    embedding_dimension: int = 768,
    seed: int = 42,
) -> dict[str, Any]:
    """Create a complete Recall@K benchmark fixture."""
    store, all_item_ids, query_relevant_ids, query_embeddings = make_test_store(
        num_items=num_items,
        embedding_dimension=embedding_dimension,
        seed=seed,
    )
    queries = [
        {"embedding": query_embeddings[q_idx], "relevant_ids": query_relevant_ids[q_idx]}
        for q_idx in range(3)
    ]

    # The expected values describe the fixture's known relevance layout rather
    # than the quality of any particular ranking model.
    expected_recall: dict[int, float] = {}
    for k in [1, 3, 5, 10]:
        scores = [
            1.0 if set(query_relevant_ids[q_idx]) & set(all_item_ids[:k]) else 0.0
            for q_idx in range(3)
        ]
        expected_recall[k] = float(np.mean(scores))

    return {
        "store": store,
        "queries": queries,
        "expected_recall": expected_recall,
        "all_item_ids": all_item_ids,
    }


def make_ndcg_benchmark_fixture(
    num_items: int = 10,
    embedding_dimension: int = 768,
    seed: int = 42,
) -> dict[str, Any]:
    """Create a complete nDCG@K benchmark fixture."""
    fixture = make_recall_benchmark_fixture(
        num_items=num_items,
        embedding_dimension=embedding_dimension,
        seed=seed,
    )

    expected_ndcg: dict[int, float] = {}
    for k in [1, 3, 5, 10]:
        scores: list[float] = []
        for q_idx in range(3):
            rel_ids = fixture["queries"][q_idx]["relevant_ids"]
            n_relevant_in_top_k = min(k, len(rel_ids))
            dcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant_in_top_k))
            ideal = min(len(rel_ids), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
            scores.append(dcg / idcg if idcg > 0 else 0.0)
        expected_ndcg[k] = float(np.mean(scores))

    return {
        "store": fixture["store"],
        "queries": fixture["queries"],
        "expected_ndcg": expected_ndcg,
        "all_item_ids": fixture["all_item_ids"],
    }
