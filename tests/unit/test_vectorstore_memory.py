# Path: tests/unit/test_vectorstore_memory.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Tests for the in-memory vector store adapter."""

from __future__ import annotations

import numpy as np
import pytest

from vlm_engineering.vectorstore.memory import MemoryVectorStore
from vlm_engineering.vectorstore.protocol import VectorStore


def _make_vector(value: float, dim: int = 4, offset: float = 0.0) -> np.ndarray:
    return np.full(dim, value + offset, dtype=np.float32)


def test_store_implements_protocol() -> None:
    store = MemoryVectorStore()
    assert isinstance(store, VectorStore)


def test_add_and_search_returns_ranked_results() -> None:
    store = MemoryVectorStore()
    items = ["a", "b", "c"]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.7, 0.7, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    store.add(embeddings, items)
    results = store.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=2)
    assert len(results) == 2
    assert results[0].item == "a"
    assert results[0].rank == 1
    assert results[1].item == "c"
    assert results[1].rank == 2


def test_search_respects_top_k() -> None:
    store = MemoryVectorStore()
    items = ["a", "b", "c", "d"]
    embeddings = np.eye(4, dtype=np.float32)
    store.add(embeddings, items)
    results = store.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=2)
    assert len(results) == 2


def test_rejects_mismatched_embeddings() -> None:
    store = MemoryVectorStore()
    with pytest.raises(ValueError, match="one vector per item"):
        store.add(
            np.array([[1.0, 0.0]], dtype=np.float32),
            ["a", "b"],
        )


def test_search_empty_store_returns_empty() -> None:
    store = MemoryVectorStore()
    assert store.search(_make_vector(1.0), top_k=5) == []


def test_upsert_replaces_existing_id() -> None:
    store = MemoryVectorStore()
    store.add(np.array([_make_vector(1.0)]), ["original"], ids=["id-1"])
    store.upsert(
        ["id-1"],
        embeddings=np.array([_make_vector(0.0, offset=2.0)], dtype=np.float32),
        items=["replaced"],
    )
    results = store.search(np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32), top_k=5)
    assert results[0].item == "replaced"
    assert store.count() == 1


def test_upsert_inserts_new_id() -> None:
    store = MemoryVectorStore()
    store.add(np.array([_make_vector(1.0)]), ["a"], ids=["id-1"])
    store.upsert(
        ["id-2"],
        embeddings=np.array([_make_vector(0.0, offset=3.0)], dtype=np.float32),
        items=["b"],
    )
    assert store.count() == 2


def test_delete_removes_item() -> None:
    store = MemoryVectorStore()
    store.add(
        np.array([_make_vector(1.0), _make_vector(0.0, offset=2.0)], dtype=np.float32),
        ["a", "b"],
        ids=["id-1", "id-2"],
    )
    store.delete(["id-1"])
    assert store.count() == 1
    results = store.search(_make_vector(1.0), top_k=5)
    assert results[0].item == "b"


def test_delete_unknown_id_is_noop() -> None:
    store = MemoryVectorStore()
    store.add(np.array([_make_vector(1.0)]), ["a"], ids=["id-1"])
    store.delete(["nonexistent"])
    assert store.count() == 1


def test_filter_by_metadata() -> None:
    from vlm_engineering.documents.schemas import VisualChunk

    chunks = [
        VisualChunk(
            "c1", "doc1", "f1.pdf", 1, "diagram", "T1", "text1", "",
            metadata={"department": "eng"},
        ),
        VisualChunk(
            "c2", "doc1", "f1.pdf", 2, "diagram", "T2", "text2", "",
            metadata={"department": "sales"},
        ),
    ]
    embeddings = np.eye(2, dtype=np.float32)
    store = MemoryVectorStore()
    store.add(embeddings, chunks, ids=["c1", "c2"])
    results = store.search(
        np.array([1.0, 0.0], dtype=np.float32), top_k=5,
        filter={"department": "eng"},
    )
    assert len(results) == 1
    assert results[0].item.chunk_id == "c1"


def test_persist_and_load_roundtrip(tmp_path) -> None:
    store = MemoryVectorStore()
    store.add(
        np.array([_make_vector(1.0), _make_vector(0.0, offset=2.0)], dtype=np.float32),
        ["a", "b"],
        ids=["id-1", "id-2"],
    )
    path = tmp_path / "store.pkl"
    store.persist(path)

    loaded = MemoryVectorStore()
    loaded.load(path)
    assert loaded.count() == 2
    results = loaded.search(_make_vector(1.0), top_k=5)
    assert results[0].item == "a"
