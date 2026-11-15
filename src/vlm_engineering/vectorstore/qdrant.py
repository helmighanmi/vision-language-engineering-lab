# Path: src/vlm_engineering/vectorstore/qdrant.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Qdrant remote vector store adapter."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..exceptions import OptionalDependencyError
from ..retrieval.in_memory import SearchResult


class QdrantVectorStore:
    """Remote vector store backed by Qdrant.

    Requires the optional ``qdrant-client`` package:

    .. code-block:: bash

        pip install "vision-language-engineering-lab[stores]"

    Chunk payloads and metadata are serialized as JSON (never pickle) into
    Qdrant point payloads. Logical chunk IDs are mapped to UUID5 identifiers
    under a fixed namespace, with the original ID preserved in the payload.
    """

    _NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # DNS namespace

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "vlm_engineering",
        api_key: str | None = None,
        https: bool = False,
        timeout: float = 30.0,
    ) -> None:
        try:
            from qdrant_client import QdrantClient  # noqa: F401
        except ImportError as exc:
            raise OptionalDependencyError(
                'Install "vision-language-engineering-lab[stores]" for Qdrant support.'
            ) from exc
        from qdrant_client import QdrantClient
        from qdrant_client.http.models import Distance, PointStruct, VectorParams

        self._QdrantClient = QdrantClient
        self._PointStruct = PointStruct
        self._VectorParams = VectorParams
        self._Distance = Distance
        self.client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key,
            https=https,
            timeout=timeout,
        )
        self.collection = collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self.client.get_collection(collection_name=self.collection)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=self._VectorParams(
                    size=0,  # lazily determined on first add
                    distance=self._Distance.COSINE,
                ),
            )

    @staticmethod
    def _logical_id_to_uuid(item_id: str) -> str:
        """Map a stable logical chunk ID to a UUID5 identifier."""
        return str(uuid.uuid5(QdrantVectorStore._NAMESPACE, item_id))

    # -- VectorStore protocol ---------------------------------------------

    def add(
        self,
        embeddings: np.ndarray,
        items: Sequence[Any],
        *,
        ids: Sequence[str] | None = None,
    ) -> None:
        from qdrant_client.http.models import PointStruct

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or len(matrix) != len(items):
            raise ValueError("embeddings must be 2-D with one vector per item.")
        resolved_ids = list(ids) if ids is not None else [f"id-{i}" for i in range(len(items))]
        if len(resolved_ids) != len(items):
            raise ValueError("ids must match the number of items.")
        existing = self.client.get_collection(collection_name=self.collection)
        existing_size = (
            existing.config.params.vectors.size
            if existing.config.params.vectors is not None
            else None
        )
        if existing_size is not None and existing_size != matrix.shape[1]:
            raise ValueError(
                f"Collection '{self.collection}' has vector size {existing_size}; "
                f"incoming embeddings have dimension {matrix.shape[1]}. "
                "Use a new collection for a new embedding schema instead of "
                "recreating an existing index."
            )
        points = []
        for item_id, item, vector in zip(resolved_ids, items, matrix, strict=True):
            payload: dict[str, Any] = {
                "chunk_id": item_id,
                "metadata": {},
            }
            if isinstance(item, dict):
                payload["metadata"] = item.get("metadata", {})
                payload["item"] = item
            else:
                payload["item"] = {"text": str(item)}
            points.append(
                PointStruct(
                    id=self._logical_id_to_uuid(item_id),
                    vector=vector.tolist(),
                    payload=payload,
                )
            )
        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query_embedding must be a 1-D vector.")
        qdrant_filter = self._build_qdrant_filter(filter) if filter else None
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query.tolist(),
            limit=top_k,
            with_payload=True,
            query_filter=qdrant_filter,
        )
        scored = []
        for point in results:
            payload = point.payload or {}
            item = payload.get("item")
            scored.append(SearchResult(item=item, score=float(point.score), rank=0))
        scored.sort(key=lambda entry: entry.score, reverse=True)
        for rank, result in enumerate(scored, start=1):
            result = SearchResult(item=result.item, score=result.score, rank=rank)
            scored[rank - 1] = result
        return scored

    @staticmethod
    def _build_qdrant_filter(filter: dict[str, Any]) -> Any:
        from qdrant_client.http.models import FieldCondition, Filter, MatchValue

        conditions = []
        for key, value in filter.items():
            conditions.append(
                FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
            )
        return Filter(must=conditions) if conditions else None

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray | None = None,
        items: Sequence[Any] | None = None,
    ) -> None:
        if embeddings is None or items is None:
            raise ValueError("Qdrant upsert requires embeddings and items.")
        self.add(embeddings, items, ids=ids)

    def delete(self, ids: Sequence[str]) -> None:
        from qdrant_client.http.models import PointIdsList

        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(ids=[self._logical_id_to_uuid(item_id) for item_id in ids]),
        )

    def count(self) -> int:
        info = self.client.get_collection(collection_name=self.collection)
        return int(info.points_count)

    def persist(self, path: str | Path) -> None:
        # Qdrant is server-side persistent; this is a no-op for compatibility.
        return None

    def load(self, path: str | Path) -> None:
        # Qdrant collections are named; this is a no-op for compatibility.
        return None

    def close(self) -> None:
        """Close the underlying Qdrant client connection."""
        try:
            self.client.close()
        except Exception:
            pass
