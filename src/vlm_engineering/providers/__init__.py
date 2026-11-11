# Path: src/vlm_engineering/providers/__init__.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Pluggable VLM, embedding and reranking providers."""

from .base import EmbeddingProvider, RerankerProvider, VLMProvider
from .qwen import QwenEmbeddingProvider, QwenRerankerProvider, QwenVLMProvider

__all__ = [
    "EmbeddingProvider",
    "QwenEmbeddingProvider",
    "QwenRerankerProvider",
    "QwenVLMProvider",
    "RerankerProvider",
    "VLMProvider",
]

try:  # pragma: no cover - optional dependency
    from .openai_compatible import (
        OpenAICompatibleEmbeddingProvider,
        OpenAICompatibleRerankerProvider,
        OpenAICompatibleVLMProvider,
    )

    __all__ += [
        "OpenAICompatibleEmbeddingProvider",
        "OpenAICompatibleRerankerProvider",
        "OpenAICompatibleVLMProvider",
    ]
except ImportError:  # pragma: no cover - optional dependency
    pass
