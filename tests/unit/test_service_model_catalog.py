# Path: tests/unit/test_service_model_catalog.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Unit tests for the configured model catalog (no backend imports)."""

from __future__ import annotations

from pathlib import Path

from vlm_engineering.service.config import ServiceConfig
from vlm_engineering.service.routers.models import build_model_catalog


def _by_capability(config: ServiceConfig) -> dict[str, str]:
    return {item.capability: item.model for item in build_model_catalog(config)}


def test_default_catalog_uses_2b_presets() -> None:
    catalog = _by_capability(ServiceConfig())
    assert catalog["generation"].endswith("Qwen3-VL-2B-Instruct")
    assert catalog["embedding"] == "Qwen/Qwen3-VL-Embedding-2B"
    assert catalog["reranking"] == "Qwen/Qwen3-VL-Reranker-2B"


def test_model_size_applies_to_every_capability() -> None:
    catalog = _by_capability(ServiceConfig(model_size="8b"))
    assert catalog["generation"].endswith("Qwen3-VL-8B-Instruct")
    assert catalog["embedding"] == "Qwen/Qwen3-VL-Embedding-8B"
    assert catalog["reranking"] == "Qwen/Qwen3-VL-Reranker-8B"


def test_generation_only_size_lists_no_retrieval_capabilities() -> None:
    catalog = _by_capability(ServiceConfig(model_size="4b"))
    assert set(catalog) == {"generation"}
    assert catalog["generation"].endswith("Qwen3-VL-4B-Instruct")


def test_explicit_model_source_applies_to_every_capability() -> None:
    catalog = _by_capability(ServiceConfig(model_id="org/custom-vl"))
    assert catalog == {
        "generation": "org/custom-vl",
        "embedding": "org/custom-vl",
        "reranking": "org/custom-vl",
    }


def test_local_model_path_is_reported_verbatim() -> None:
    path = Path("/models/qwen-local")
    catalog = _by_capability(ServiceConfig(model_path=path))
    assert catalog["generation"] == str(path)
    assert catalog["embedding"] == str(path)


def test_openai_compatible_catalog_keeps_qwen_retrieval_models() -> None:
    config = ServiceConfig(
        vlm_provider="openai-compatible",
        vlm_base_url="http://localhost:8000",
        vlm_model="served-vlm",
    )
    models = {item.capability: item for item in build_model_catalog(config)}
    assert models["generation"].model == "served-vlm"
    assert models["generation"].provider == "openai-compatible"
    assert models["embedding"].provider == "qwen"
    assert models["embedding"].model == "Qwen/Qwen3-VL-Embedding-2B"
