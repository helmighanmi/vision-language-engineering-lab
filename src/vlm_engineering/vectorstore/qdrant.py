# Path: src/vlm_engineering/vectorstore/qdrant.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Qdrant remote vector store adapter."""

from __future__ import annotations

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

    Items and metadata are pickled into a Qdrant payload so results preserve
    the original payload.
    """

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

    # -- VectorStore protocol ---------------------------------------------

    def add(
        self,
        embeddings: np.ndarray,
        items: Sequence[Any],
        *,
        ids: Sequence[str] | None = None,
    ) -> None:
        import pickle

        from qdrant_client.http.models import PointStruct, VectorParams

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or len(matrix) != len(items):
            raise ValueError("embeddings must be 2-D with one vector per item.")
        resolved_ids = list(ids) if ids is not None else [f"id-{i}" for i in range(len(items))]
        if len(resolved_ids) != len(items):
            raise ValueError("ids must match the number of items.")
        # Recreate collection with correct dimension if needed.
        existing = self.client.get_collection(collection_name=self.collection)
        if existing.config.params.vectors is None or existing.config.params.vectors.size != matrix.shape[1]:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=matrix.shape[1],
                    distance=self._Distance.COSINE,
                ),
            )
        points = []
        for item_id, item, vector in zip(resolved_ids, items, matrix, strict=True):
            points.append(
                PointStruct(
                    id=item_id,
                    vector=vector.tolist(),
                    payload={"item": pickle.dumps(item).hex(), "metadata": {}},
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
        import pickle

        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 1:
            raise ValueError("query_embedding must be a 1-D vector.")
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query.tolist(),
            limit=top_k,
            with_payload=True,
        )
        scored = []
        for point in results:
            payload = point.payload or {}
            item_blob = payload.get("item")
            item = pickle.loads(bytes.fromhex(item_blob)) if item_blob else None
            scored.append(SearchResult(item=item, score=float(point.score), rank=0))
        scored.sort(key=lambda entry: entry.score, reverse=True)
        for rank, result in enumerate(scored, start=1):
            result = SearchResult(item=result.item, score=result.score, rank=rank)
            scored[rank - 1] = result
        return scored

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
            points_selector=PointIdsList(ids=list(ids)),
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
