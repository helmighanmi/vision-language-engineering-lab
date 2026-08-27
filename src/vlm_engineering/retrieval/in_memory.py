# Path: src/vlm_engineering/retrieval/in_memory.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Small dependency-light cosine vector index for demos and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class SearchResult:
    item: Any
    score: float
    rank: int


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._items: list[Any] = []

    def add(self, embeddings: np.ndarray, items: Sequence[Any]) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or len(matrix) != len(items):
            raise ValueError("embeddings must be 2-D with one vector per item.")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-12, None)
        self._embeddings = matrix
        self._items = list(items)

    def search(self, query_embedding: np.ndarray, *, top_k: int = 5) -> list[SearchResult]:
        if self._embeddings is None:
            raise RuntimeError("Index is empty.")
        vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        scores = self._embeddings @ vector
        order = np.argsort(-scores)[: max(1, top_k)]
        return [
            SearchResult(item=self._items[int(idx)], score=float(scores[idx]), rank=rank)
            for rank, idx in enumerate(order, start=1)
        ]
