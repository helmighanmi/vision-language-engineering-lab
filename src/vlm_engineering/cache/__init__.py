# Path: src/vlm_engineering/cache/__init__.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Simple disk-backed cache for embeddings and answers."""

from .cache import DiskCache, MemoryCache

__all__ = ["DiskCache", "MemoryCache"]
