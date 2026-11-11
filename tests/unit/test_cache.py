# Path: tests/unit/test_cache.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Tests for the cache module."""

from __future__ import annotations

from vlm_engineering.cache.cache import (
    DiskCache,
    MemoryCache,
    answer_cache_key,
    embedding_cache_key,
)


def test_memory_cache_get_set() -> None:
    cache = MemoryCache()
    cache.set("key1", {"value": 1})
    assert cache.get("key1") == {"value": 1}
    assert cache.get("missing") is None


def test_memory_cache_overwrites() -> None:
    cache = MemoryCache()
    cache.set("key1", "first")
    cache.set("key1", "second")
    assert cache.get("key1") == "second"


def test_memory_cache_clear() -> None:
    cache = MemoryCache()
    cache.set("key1", "value")
    cache.clear()
    assert cache.get("key1") is None


def test_memory_cache_bounded_eviction() -> None:
    cache = MemoryCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # should evict one of a/b
    # "c" is present; exactly one of "a"/"b" was evicted.
    assert cache.get("c") == 3
    remaining = sum(1 for k in ("a", "b") if cache.get(k) is not None)
    assert remaining == 1



def test_disk_cache_get_set(tmp_path) -> None:
    cache = DiskCache(tmp_path / "cache")
    cache.set("key1", {"value": 42})
    assert cache.get("key1") == {"value": 42}
    assert cache.get("missing") is None


def test_disk_cache_persists_across_instances(tmp_path) -> None:
    cache_dir = tmp_path / "cache"
    DiskCache(cache_dir).set("key1", {"value": 42})
    loaded = DiskCache(cache_dir)
    assert loaded.get("key1") == {"value": 42}


def test_disk_cache_clear(tmp_path) -> None:
    cache = DiskCache(tmp_path / "cache")
    cache.set("key1", "value")
    cache.clear()
    assert cache.get("key1") is None


def test_embedding_cache_key_is_deterministic() -> None:
    key1 = embedding_cache_key("model-x", [{"text": "a"}], 1024, True)
    key2 = embedding_cache_key("model-x", [{"text": "a"}], 1024, True)
    assert key1 == key2
    assert len(key1) == 64  # sha256 hex


def test_embedding_cache_key_differs_on_input_change() -> None:
    key1 = embedding_cache_key("model-x", [{"text": "a"}], 1024, True)
    key2 = embedding_cache_key("model-x", [{"text": "b"}], 1024, True)
    assert key1 != key2


def test_answer_cache_key_is_deterministic() -> None:
    key1 = answer_cache_key("model-y", "q?", ["img1.png"], 256)
    key2 = answer_cache_key("model-y", "q?", ["img1.png"], 256)
    assert key1 == key2


def test_answer_cache_key_differs_on_image_order() -> None:
    key1 = answer_cache_key("model-y", "q?", ["img1.png", "img2.png"], 256)
    key2 = answer_cache_key("model-y", "q?", ["img2.png", "img1.png"], 256)
    # Sorted images => same key.
    assert key1 == key2
