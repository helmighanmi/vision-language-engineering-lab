# Path: src/vlm_engineering/providers/base.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Provider protocols for interchangeable VLM, embedding and reranking backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

from ..inputs import MultimodalInput, normalize_input, normalize_inputs


@runtime_checkable
class VLMProvider(Protocol):
    """Generate text from images and/or text prompts."""

    def generate(
        self,
        images: Sequence[str | Path],
        prompt: str,
        *,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        """Generate from one or more images plus a text prompt."""
        ...

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        """Generate from a text prompt alone (no images)."""
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Encode text/image/mixed inputs into vectors."""

    def encode(
        self,
        inputs: Sequence[MultimodalInput],
        *,
        prompt: str | None = None,
    ) -> np.ndarray:
        """Encode a sequence of multimodal inputs into vectors."""
        ...

    def embed_text(self, text: str) -> np.ndarray:
        """Encode a single text input into a vector."""
        ...

    def embed_image(self, image: str | Path) -> np.ndarray:
        """Encode a single local image into a vector."""
        ...


@runtime_checkable
class RerankerProvider(Protocol):
    """Score query/document pairs for reranking."""

    def score(
        self,
        query: MultimodalInput,
        documents: Sequence[MultimodalInput],
        *,
        prompt: str | None = None,
    ) -> np.ndarray:
        """Return one score per document."""
        ...

    def rerank(
        self,
        query: MultimodalInput,
        documents: Sequence[MultimodalInput],
        *,
        top_k: int | None = None,
        prompt: str | None = None,
    ) -> list[Any]:
        """Return ranked documents with scores."""
        ...


def normalize_provider_input(value: MultimodalInput) -> dict[str, str]:
    """Normalize a multimodal input into a text/image dict."""
    return normalize_input(value)


def normalize_provider_inputs(values: Sequence[MultimodalInput]) -> list[dict[str, str]]:
    """Normalize a sequence of multimodal inputs."""
    return normalize_inputs(values)
