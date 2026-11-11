# Path: src/vlm_engineering/vectorstore/protocol.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Protocol for interchangeable vector stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from ..retrieval.in_memory import SearchResult


@runtime_checkable
class VectorStore(Protocol):
    """A vector store that indexes items and returns ranked search results.

    Implementations must be deterministic for the same inputs and must preserve
    the original item (its full payload) in each ``SearchResult``.
    """

    def add(
        self,
        embeddings: np.ndarray,
        items: Sequence[Any],
        *,
        ids: Sequence[str] | None = None,
    ) -> None:
        """Index ``embeddings`` alongside ``items``.

        Args:
            embeddings:
                2-D array with one vector per item.
            items:
                Original payloads preserved verbatim in search results.
            ids:
                Optional stable identifiers. When provided, later ``upsert`` /
                ``delete`` calls can target them. Generated deterministically
                when omitted.
        """
        ...

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return the top ``top_k`` items most similar to ``query_embedding``.

        Args:
            query_embedding:
                1-D vector whose dimension must match the indexed vectors.
            top_k:
                Maximum number of results.
            filter:
                Optional metadata filter applied before ranking. Keys match
                item ``metadata`` fields; values are matched exactly.
        """
        ...

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray | None = None,
        items: Sequence[Any] | None = None,
    ) -> None:
        """Replace or insert the vectors/items identified by ``ids``.

        Missing ids are inserted; existing ids are replaced atomically per id.
        """
        ...

    def delete(self, ids: Sequence[str]) -> None:
        """Remove the vectors/items identified by ``ids``."""
        ...

    def count(self) -> int:
        """Return the number of indexed items."""
        ...

    def persist(self, path: str | Path) -> None:
        """Write the store to ``path`` for later :meth:`load`."""
        ...

    def load(self, path: str | Path) -> None:
        """Replace this store's contents from a previously persisted ``path``."""
        ...
