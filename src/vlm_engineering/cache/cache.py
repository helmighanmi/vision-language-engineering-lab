# Path: src/vlm_engineering/cache/cache.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Simple caches for embeddings and generated answers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


class MemoryCache:
    """In-memory key/value cache with bounded size."""

    def __init__(self, max_size: int = 10000) -> None:
        self._store: dict[str, Any] = {}
        self._max_size = max_size

    def _key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        return self._store.get(self._key(key))

    def set(self, key: str, value: Any) -> None:
        if len(self._store) >= self._max_size and self._key(key) not in self._store:
            # Drop one arbitrary entry to stay bounded.
            self._store.pop(next(iter(self._store)))
        self._store[self._key(key)] = value

    def clear(self) -> None:
        self._store.clear()


class DiskCache:
    """Disk-backed JSON cache for embeddings and answers."""

    def __init__(self, directory: str | Path = ".cache/vlm_engineering") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def clear(self) -> None:
        for path in self.directory.glob("*.json"):
            path.unlink()


def embedding_cache_key(
    model_source: str,
    inputs: Sequence[dict[str, str]],
    dimensions: int | None,
    normalize: bool,
) -> str:
    """Stable cache key for an embedding batch."""
    payload = {
        "model": model_source,
        "inputs": inputs,
        "dimensions": dimensions,
        "normalize": normalize,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def answer_cache_key(
    model_source: str,
    question: str,
    image_refs: Sequence[str],
    max_new_tokens: int,
) -> str:
    """Stable cache key for a generated answer."""
    payload = {
        "model": model_source,
        "question": question,
        "images": sorted(image_refs),
        "max_new_tokens": max_new_tokens,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
