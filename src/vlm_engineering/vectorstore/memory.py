# Path: src/vlm_engineering/vectorstore/memory.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""In-memory vector store adapter backed by the deterministic cosine index."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..documents.schemas import VisualChunk
from ..retrieval.in_memory import InMemoryVectorIndex, SearchResult


class MemoryVectorStore:
    """In-memory vector store with optional disk persistence.

    This is a thin adapter over
    :class:`~vlm_engineering.retrieval.in_memory.InMemoryVectorIndex`.
    Vectors are normalized at index time; search returns cosine similarity.
    """

    def __init__(self) -> None:
        self._index = InMemoryVectorIndex()
        self._items: list[Any] = []
        self._embeddings: np.ndarray | None = None
        self._id_to_position: dict[str, int] = {}
        self._position_to_id: list[str | None] = []
        self._metadata: list[dict[str, Any]] = []

    # -- internal helpers -------------------------------------------------

    def _default_ids(self, count: int) -> list[str]:
        return [f"id-{len(self._items) + index}" for index in range(count)]

    def _extract_metadata(self, item: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if isinstance(item, VisualChunk):
            metadata = {
                "document_id": item.document_id,
                "source_file": item.source_file,
                "page": item.page,
                "chunk_type": item.chunk_type,
                "chunk_id": item.chunk_id,
            }
        if hasattr(item, "metadata") and isinstance(item.metadata, dict):
            metadata = {**metadata, **item.metadata}
        return metadata

    def _apply_filter(
        self,
        results: list[SearchResult],
        filter: dict[str, Any] | None,
    ) -> list[SearchResult]:
        if not filter:
            return results
        kept = []
        for result in results:
            metadata = self._metadata_for_item(result.item)
            if all(metadata.get(key) == value for key, value in filter.items()):
                kept.append(result)
        return kept

    def _metadata_for_item(self, item: Any) -> dict[str, Any]:
        if item in self._items:
            return self._metadata[self._items.index(item)]
        return {}

    def _rebuild_index(self) -> None:
        """Recreate the underlying index from surviving items/embeddings."""
        if self._embeddings is None or len(self._embeddings) == 0:
            self._index = InMemoryVectorIndex()
            return
        surviving = [
            (item, vector)
            for item, vector in zip(self._items, self._embeddings, strict=True)
            if item is not None
        ]
        if not surviving:
            self._index = InMemoryVectorIndex()
            return
        items, vectors = zip(*surviving, strict=True)
        matrix = np.asarray(vectors, dtype=np.float64)
        self._index = InMemoryVectorIndex()
        self._index.add(matrix, list(items))

    def _compact_after_delete(self) -> None:
        """Remove None slots and rebuild id/position mappings."""
        surviving_items: list[Any] = []
        surviving_embeddings: list[np.ndarray] = []
        surviving_metadata: list[dict[str, Any]] = []
        new_id_to_position: dict[str, int] = {}
        for position, item in enumerate(self._items):
            if item is None:
                continue
            new_position = len(surviving_items)
            item_id = self._position_to_id[position]
            if item_id is not None:
                new_id_to_position[item_id] = new_position
            surviving_items.append(item)
            if self._embeddings is not None:
                surviving_embeddings.append(self._embeddings[position])
            surviving_metadata.append(self._metadata[position])
        self._items = surviving_items
        self._metadata = surviving_metadata
        self._id_to_position = new_id_to_position
        if surviving_embeddings:
            self._embeddings = np.asarray(surviving_embeddings, dtype=np.float64)
        else:
            self._embeddings = None
        self._position_to_id = [None] * len(self._items)
        for item_id, new_position in new_id_to_position.items():
            self._position_to_id[new_position] = item_id
        self._rebuild_index()

    # -- VectorStore protocol ---------------------------------------------

    def add(
        self,
        embeddings: np.ndarray,
        items: Sequence[Any],
        *,
        ids: Sequence[str] | None = None,
    ) -> None:
        matrix = np.asarray(embeddings, dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) != len(items):
            raise ValueError("embeddings must be 2-D with one vector per item.")
        resolved_ids = list(ids) if ids is not None else self._default_ids(len(items))
        if len(resolved_ids) != len(items):
            raise ValueError("ids must match the number of items.")
        duplicate = set(resolved_ids) & set(self._id_to_position)
        if duplicate:
            raise ValueError(f"ids already exist: {sorted(duplicate)}.")
        start = len(self._items)
        self._items.extend(items)
        for offset, item_id in enumerate(resolved_ids):
            position = start + offset
            self._id_to_position[item_id] = position
            self._position_to_id.append(item_id)
            self._metadata.append(self._extract_metadata(items[offset]))
        if self._embeddings is None:
            self._embeddings = matrix.copy()
        else:
            self._embeddings = np.vstack([self._embeddings, matrix])
        self._rebuild_index()

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if self._embeddings is None or len(self._embeddings) == 0:
            return []
        results = self._index.search(query_embedding, top_k=max(top_k, self.count()))
        results = self._apply_filter(results, filter)
        return results[:top_k]

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray | None = None,
        items: Sequence[Any] | None = None,
    ) -> None:
        ids = list(ids)
        embeddings = (
            None if embeddings is None else np.asarray(embeddings, dtype=np.float64)
        )
        items = None if items is None else list(items)
        if embeddings is not None and items is not None:
            if len(ids) != len(embeddings) or len(ids) != len(items):
                raise ValueError("ids, embeddings and items must have equal length.")
        for item_id in ids:
            position = self._id_to_position.get(item_id)
            row_index = ids.index(item_id)
            if position is None:
                # Insert a new id.
                position = len(self._items)
                row = (
                    embeddings[row_index].reshape(1, -1)
                    if embeddings is not None
                    else None
                )
                item = items[row_index] if items is not None else None
                self._items.append(item)
                self._id_to_position[item_id] = position
                self._position_to_id.append(item_id)
                self._metadata.append(self._extract_metadata(item) if item is not None else {})
                if row is not None:
                    if self._embeddings is None:
                        self._embeddings = row.copy()
                    else:
                        self._embeddings = np.vstack([self._embeddings, row])
            else:
                if embeddings is not None:
                    vector = embeddings[row_index].reshape(1, -1)
                    if self._embeddings is not None:
                        self._embeddings[position] = vector / np.linalg.norm(vector)
                if items is not None:
                    self._items[position] = items[row_index]
                    self._metadata[position] = self._extract_metadata(
                        items[row_index]
                    )
        self._rebuild_index()

    def delete(self, ids: Sequence[str]) -> None:
        for item_id in ids:
            position = self._id_to_position.pop(item_id, None)
            if position is None:
                continue
            self._items[position] = None
            if position < len(self._position_to_id):
                self._position_to_id[position] = None
            self._metadata[position] = {}
        self._compact_after_delete()

    def count(self) -> int:
        return sum(1 for item in self._items if item is not None)

    def persist(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "items": self._items,
            "embeddings": (
                self._embeddings if self._embeddings is not None else None
            ),
            "id_to_position": self._id_to_position,
            "position_to_id": self._position_to_id,
            "metadata": self._metadata,
        }
        destination.write_bytes(pickle.dumps(payload))

    def load(self, path: str | Path) -> None:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Vector store not found: {source}")
        payload = pickle.loads(source.read_bytes())
        self._items = list(payload["items"])
        self._metadata = list(payload["metadata"])
        embeddings = payload["embeddings"]
        self._embeddings = (
            np.asarray(embeddings, dtype=np.float64) if embeddings is not None else None
        )
        self._id_to_position = dict(payload.get("id_to_position", {}))
        self._position_to_id = list(payload.get("position_to_id", [None] * len(self._items)))
        self._rebuild_index()
