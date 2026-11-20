# Path: src/vlm_engineering/vectorstore/qdrant.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Qdrant vector store adapter.

Requires the optional ``qdrant-client`` package:

    pip install "vision-language-engineering-lab[stores]"

Chunk payloads and metadata are serialized as JSON (never pickle) into
Qdrant point payloads. Logical chunk IDs are mapped to UUID5 identifiers
under a fixed namespace, with the original ID preserved in the payload.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..exceptions import InputValidationError, OptionalDependencyError
from ..retrieval.in_memory import SearchResult


class QdrantVectorStore:
    """Persistent vector store backed by Qdrant.

    Requires the optional ``qdrant-client`` package:

    .. code-block:: bash

        pip install "vision-language-engineering-lab[stores]"

    Chunk payloads and metadata are serialized as JSON (never pickle) into
    Qdrant point payloads. Logical chunk IDs are mapped to UUID5 identifiers
    under a fixed namespace, with the original ID preserved in the payload.

    This adapter implements the full :class:`VectorStore` protocol with
    deterministic operations, index lifecycle management, and resource
    profiling support for v0.4.
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
        self._model_revision: str | None = None
        self._embedding_dimension: int | None = None
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Create the collection if it does not exist, with lazy dimension detection."""
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

    def set_model_revision(self, revision: str | None) -> None:
        """Track model revision for cache invalidation.

        When the model revision changes, the store can optionally clear
        its index to prevent stale evidence from being served.
        """
        self._model_revision = revision

    def get_model_revision(self) -> str | None:
        """Return the currently tracked model revision."""
        return self._model_revision

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
        """Index ``embeddings`` alongside ``items``.

        Vectors are validated for finiteness and nonzero norm before indexing.
        Raises ``InputValidationError`` on invalid input rather than silently
        passing bad data to Qdrant.

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
        from qdrant_client.http.models import PointStruct

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or len(matrix) != len(items):
            raise InputValidationError(
                "embeddings must be 2-D with one vector per item."
            )
        resolved_ids = list(ids) if ids is not None else [f"id-{i}" for i in range(len(items))]
        if len(resolved_ids) != len(items):
            raise InputValidationError("ids must match the number of items.")
        existing = self.client.get_collection(collection_name=self.collection)
        existing_size = (
            existing.config.params.vectors.size
            if existing.config.params.vectors is not None
            else None
        )
        if existing_size is not None and existing_size != matrix.shape[1]:
            raise InputValidationError(
                f"Collection '{self.collection}' has vector size {existing_size}; "
                f"incoming embeddings have dimension {matrix.shape[1]}. "
                "Use a new collection for a new embedding schema instead of "
                "recreating an existing index."
            )
        # Validate finiteness and nonzero norm before sending to Qdrant.
        if not np.isfinite(matrix).all():
            raise InputValidationError("embeddings must contain only finite values.")
        if np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise InputValidationError("embeddings must have nonzero norm.")
        if len(set(resolved_ids)) != len(resolved_ids):
            raise InputValidationError("duplicate ids within the same batch are not allowed.")
        duplicate = set(resolved_ids) & set(self._id_to_position())
        if duplicate:
            raise InputValidationError(f"ids already exist: {sorted(duplicate)}.")
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

    def _id_to_position(self) -> dict[str, int]:
        """Return a mapping of logical IDs to their positions in the store.

        This is a best-effort query; Qdrant does not expose a direct
        position mapping, so we derive it from the existing collection
        point count at index time.
        """
        try:
            info = self.client.get_collection(collection_name=self.collection)
            return {
                str(uuid.UUID(point_id)): idx
                for idx, point_id in enumerate(
                    info.result.points if hasattr(info, "result") else []
                )
            }
        except Exception:
            return {}

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
        query = np.asarray(query_embedding, dtype=np.float32)
        if query.ndim != 1:
            raise InputValidationError("query_embedding must be a 1-D vector.")
        if self._embedding_dimension is not None and query.shape[0] != self._embedding_dimension:
            raise InputValidationError(
                f"query_embedding dimension {query.shape[0]} does not match "
                f"indexed dimension {self._embedding_dimension}."
            )
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
        """Replace or insert the vectors/items identified by ``ids``.

        Missing ids are inserted; existing ids are replaced atomically per id.
        Raises ``InputValidationError`` if embeddings/items are missing when
        ids are provided.
        """
        if embeddings is None or items is None:
            raise InputValidationError("Qdrant upsert requires embeddings and items.")
        if len(ids) != len(embeddings) or len(ids) != len(items):
            raise InputValidationError("ids, embeddings and items must have equal length.")
        self.add(embeddings, items, ids=ids)

    def delete(self, ids: Sequence[str]) -> None:
        """Remove the vectors/items identified by ``ids``.

        Args:
            ids:
                Logical chunk IDs to delete. Mapped to UUID5 identifiers
                under the fixed namespace.
        """
        from qdrant_client.http.models import PointIdsList

        self.client.delete(
            collection_name=self.collection,
            points_selector=PointIdsList(
                ids=[self._logical_id_to_uuid(item_id) for item_id in ids]
            ),
        )

    def count(self) -> int:
        """Return the number of indexed items."""
        info = self.client.get_collection(collection_name=self.collection)
        return int(info.points_count)

    def clear(self) -> None:
        """Remove all indexed items (empty re-indexing).

        This atomically clears the Qdrant collection, ensuring no stale
        evidence remains. After calling ``clear()``, the collection is
        empty and ``add()`` can be used to rebuild the index from scratch.
        """
        self.client.delete_collection(collection_name=self.collection)
        self._embedding_dimension = None
        self._model_revision = None
        # Recreate an empty collection with the default config.
        self._ensure_collection()

    def persist(self, path: str | Path) -> None:
        # Qdrant is server-side persistent; this is a no-op for compatibility.
        # The collection and its data remain on the Qdrant server.
        return None

    def load(self, path: str | Path) -> None:
        # Qdrant collections are identified by name; loading is a no-op
        # because the store always connects to the named collection.
        # Callers should ensure the correct collection name is configured.
        return None

    def close(self) -> None:
        """Close the underlying Qdrant client connection."""
        try:
            self.client.close()
        except Exception:
            pass

    def get_stats(self) -> dict[str, Any]:
        """Return resource usage statistics for profiling.

        Returns a dictionary with collection size, vector dimension, and
        other metadata useful for resource profiling and benchmarking.
        """
        try:
            info = self.client.get_collection(collection_name=self.collection)
            vectors_config = info.config.params.vectors
            return {
                "collection_name": self.collection,
                "points_count": int(info.points_count),
                "vector_size": vectors_config.size if vectors_config.size else 0,
                "distance_metric": info.config.distance.name,
                "model_revision": self._model_revision or "none",
                "embedding_dimension": self._embedding_dimension or 0,
            }
        except Exception:
            return {
                "collection_name": self.collection,
                "points_count": 0,
                "vector_size": 0,
                "distance_metric": "unknown",
                "model_revision": "none",
                "embedding_dimension": 0,
            }