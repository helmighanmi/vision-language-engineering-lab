# Path: src/vlm_engineering/vectorstore/__init__.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Persistent vector stores for multimodal retrieval.

The :class:`VectorStore` protocol keeps the existing :class:`SearchResult`
contract so memory, sqlite-vec, and Qdrant adapters are interchangeable.
"""

from .memory import MemoryVectorStore
from .protocol import VectorStore

__all__ = [
    "VectorStore",
    "MemoryVectorStore",
]

try:  # pragma: no cover - optional dependency
    from .sqlite_vec import SqliteVecVectorStore

    __all__.append("SqliteVecVectorStore")
except ImportError:  # pragma: no cover - optional dependency
    SqliteVecVectorStore = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .qdrant import QdrantVectorStore

    __all__.append("QdrantVectorStore")
except ImportError:  # pragma: no cover - optional dependency
    QdrantVectorStore = None  # type: ignore
