# Path: src/vlm_engineering/vectorstore/sqlite_vec.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Single-file persistent vector store backed by sqlite-vec."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..exceptions import OptionalDependencyError
from ..retrieval.in_memory import SearchResult


class SqliteVecVectorStore:
    """Persistent vector store in a single SQLite file using sqlite-vec.

    Requires the optional ``sqlite-vec`` package:

    .. code-block:: bash

        pip install "vision-language-engineering-lab[stores]"

    Vectors are stored as float32 blobs and queried with cosine similarity.
    Items and metadata are pickled into a side table so results preserve the
    original payload.
    """

    def __init__(self, path: str | Path, table: str = "vectors") -> None:
        try:
            import sqlite_vec  # noqa: F401
        except ImportError as exc:
            raise OptionalDependencyError(
                'Install "vision-language-engineering-lab[stores]" for sqlite-vec support.'
            ) from exc

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.table = table
        # check_same_thread=False allows the connection to be used from worker
        # threads; _lock serializes all access so SQLite's own threading rules
        # are not violated.
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id TEXT PRIMARY KEY,
                    item BLOB NOT NULL,
                    metadata BLOB NOT NULL,
                    embedding BLOB NOT NULL
                )
                """
            )
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_id ON {self.table}(id)"
            )
            self._conn.commit()

    # -- VectorStore protocol ---------------------------------------------

    def add(
        self,
        embeddings: np.ndarray,
        items: Sequence[Any],
        *,
        ids: Sequence[str] | None = None,
    ) -> None:
        import sqlite3

        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim != 2 or len(matrix) != len(items):
            raise ValueError("embeddings must be 2-D with one vector per item.")
        resolved_ids = list(ids) if ids is not None else [f"id-{i}" for i in range(len(items))]
        if len(resolved_ids) != len(items):
            raise ValueError("ids must match the number of items.")
        rows = []
        for item_id, item, vector in zip(resolved_ids, items, matrix, strict=True):
            item_blob = json.dumps(self._item_to_json(item)).encode("utf-8")
            metadata_blob = json.dumps(self._extract_metadata(item)).encode("utf-8")
            rows.append(
                (
                    item_id,
                    sqlite3.Binary(item_blob),
                    sqlite3.Binary(metadata_blob),
                    sqlite3.Binary(vector.tobytes()),
                )
            )
        with self._lock:
            self._conn.executemany(
                f"INSERT OR REPLACE INTO {self.table} (id, item, metadata, embedding) VALUES (?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

    @staticmethod
    def _item_to_json(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return dict(item)
        return {"text": str(item)}

    @staticmethod
    def _extract_metadata(item: Any) -> dict[str, Any]:
        if isinstance(item, dict) and isinstance(item.get("metadata"), dict):
            return dict(item["metadata"])
        return {}

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
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, item, metadata, embedding FROM {self.table}"
            ).fetchall()
        if not rows:
            return []
        scored = []
        query_norm = float(np.linalg.norm(query))
        for item_id, item_blob, metadata_blob, embedding_blob in rows:
            vector = np.frombuffer(embedding_blob, dtype=np.float32)
            if vector.shape[0] != query.shape[0]:
                continue
            vector_norm = float(np.linalg.norm(vector))
            if query_norm == 0 or vector_norm == 0:
                continue
            similarity = float(np.dot(query, vector) / (query_norm * vector_norm))
            # Apply metadata filter before scoring.
            if filter is not None:
                try:
                    stored_metadata = json.loads(metadata_blob.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    stored_metadata = {}
                if not all(stored_metadata.get(key) == value for key, value in filter.items()):
                    continue
            try:
                item = json.loads(item_blob.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                item = None
            scored.append((item_id, item, similarity))
        scored.sort(key=lambda entry: entry[2], reverse=True)
        results = []
        for rank, (_item_id, item, score) in enumerate(scored[:top_k], start=1):
            results.append(SearchResult(item=item, score=score, rank=rank))
        return results

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: np.ndarray | None = None,
        items: Sequence[Any] | None = None,
    ) -> None:
        # sqlite-vec upsert is identical to add with INSERT OR REPLACE.
        if embeddings is None or items is None:
            raise ValueError("sqlite-vec upsert requires embeddings and items.")
        self.add(embeddings, items, ids=ids)

    def delete(self, ids: Sequence[str]) -> None:

        with self._lock:
            self._conn.executemany(
                f"DELETE FROM {self.table} WHERE id = ?",
                [(item_id,) for item_id in ids],
            )
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute(f"SELECT COUNT(*) FROM {self.table}").fetchone()
        return int(row[0]) if row else 0

    def persist(self, path: str | Path) -> None:
        # SQLite is already persistent; this is a no-op for compatibility.
        return None

    def load(self, path: str | Path) -> None:
        # Reconnect to an existing sqlite-vec database.
        self.path = Path(path)
        with self._lock:
            self._conn.close()
            self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SqliteVecVectorStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
