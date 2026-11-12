# Path: src/vlm_engineering/service/catalog.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Resolve configured model labels without importing or loading backends."""

from __future__ import annotations

from ..qwen.registry import resolve_qwen_model_id
from .config import ServiceConfig

RETRIEVAL_SIZES = frozenset({"2b", "8b"})


def generation_model(config: ServiceConfig) -> str:
    """Return the model label the generation route would use."""
    if config.vlm_provider == "openai-compatible":
        return str(config.vlm_model)
    return _qwen_model(config)


def embedding_model(config: ServiceConfig) -> str | None:
    """Return the embedding label, or ``None`` when the preset is unsupported."""
    return _retrieval_model(config, "Embedding")


def reranking_model(config: ServiceConfig) -> str | None:
    """Return the reranking label, or ``None`` when the preset is unsupported."""
    return _retrieval_model(config, "Reranker")


def _qwen_model(config: ServiceConfig) -> str:
    if config.model_path is not None:
        return str(config.model_path)
    if config.model_id is not None:
        return config.model_id
    return resolve_qwen_model_id(model_size=config.model_size)


def _retrieval_model(config: ServiceConfig, kind: str) -> str | None:
    if config.model_path is not None:
        return str(config.model_path)
    if config.model_id is not None:
        return config.model_id
    if config.model_size is not None and config.model_size not in RETRIEVAL_SIZES:
        # 4b is generation-only; the retrieval wrappers reject it.
        return None
    size = config.model_size or "2b"
    return f"Qwen/Qwen3-VL-{kind}-{size.upper()}"
