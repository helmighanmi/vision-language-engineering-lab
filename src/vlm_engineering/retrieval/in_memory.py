# Path: src/vlm_engineering/retrieval/in_memory.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Small deterministic cosine index; each add atomically replaces the collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..exceptions import InputValidationError
from ..inputs import positive_int


@dataclass(frozen=True, slots=True)
class SearchResult:
    item: Any
    score: float
    rank: int


class InMemoryVectorIndex:
    def __init__(self) -> None:
        self._embeddings: np.ndarray | None = None
        self._items: list[Any] = []

    def clear(self) -> None:
        self._embeddings = None
        self._items = []

    def add(self, embeddings: np.ndarray, items: Sequence[Any]) -> None:
        matrix = np.asarray(embeddings, dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) != len(items):
            raise InputValidationError("embeddings must be 2-D with one vector per item.")
        if not len(items):
            self.clear()
            return
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if not np.isfinite(matrix).all() or not np.isfinite(norms).all() or np.any(norms == 0):
            raise InputValidationError("Index vectors must be finite and have nonzero norm.")
        normalized = matrix / norms
        self._embeddings = normalized
        self._items = list(items)

    def search(self, query_embedding: np.ndarray, *, top_k: int = 5) -> list[SearchResult]:
        positive_int(top_k, "top_k")
        if self._embeddings is None:
            raise RuntimeError("Index is empty.")
        vector = np.asarray(query_embedding, dtype=np.float64)
        if vector.ndim != 1 or vector.shape[0] != self._embeddings.shape[1]:
            raise InputValidationError("Query vector dimension must match the index.")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(vector).all() or not np.isfinite(norm) or norm == 0:
            raise InputValidationError("Query vector must be finite and have nonzero norm.")
        scores = self._embeddings @ (vector / norm)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            SearchResult(self._items[int(idx)], float(scores[idx]), rank)
            for rank, idx in enumerate(order, start=1)
        ]
