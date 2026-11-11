# Path: src/vlm_engineering/providers/qwen.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Qwen3-VL providers backed by the package's own Qwen wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..inputs import MultimodalInput, normalize_input, normalize_inputs
from ..qwen.model import QwenVLModel
from ..retrieval import QwenVLEmbedder, QwenVLReranker


class QwenVLMProvider:
    """Qwen3-VL generative provider implementing :class:`VLMProvider`."""

    def __init__(self, model: QwenVLModel | None = None, **model_kwargs: Any) -> None:
        self.model = model or QwenVLModel(**model_kwargs)

    def generate(
        self,
        images: Sequence[str | Path],
        prompt: str,
        *,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        return self.model.generate_images(
            images,
            prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        return self.model.generate_text(
            prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
        )

    def unload(self) -> None:
        self.model.unload()


class QwenEmbeddingProvider:
    """Qwen3-VL-Embedding provider implementing :class:`EmbeddingProvider`."""

    def __init__(self, model: QwenVLEmbedder | None = None, **model_kwargs: Any) -> None:
        self.model = model or QwenVLEmbedder(**model_kwargs)

    def encode(
        self,
        inputs: Sequence[MultimodalInput],
        *,
        prompt: str | None = None,
    ) -> np.ndarray:
        prepared = normalize_inputs(inputs)
        return self.model.encode(prepared, prompt=prompt)

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.embed_text(text)

    def embed_image(self, image: str | Path) -> np.ndarray:
        return self.model.embed_image(image)

    def unload(self) -> None:
        self.model.unload()


class QwenRerankerProvider:
    """Qwen3-VL-Reranker provider implementing :class:`RerankerProvider`."""

    def __init__(self, model: QwenVLReranker | None = None, **model_kwargs: Any) -> None:
        self.model = model or QwenVLReranker(**model_kwargs)

    def score(
        self,
        query: MultimodalInput,
        documents: Sequence[MultimodalInput],
        *,
        prompt: str | None = None,
    ) -> np.ndarray:
        return self.model.score(
            normalize_input(query),
            normalize_inputs(documents),
            prompt=prompt,
        )

    def rerank(
        self,
        query: MultimodalInput,
        documents: Sequence[MultimodalInput],
        *,
        top_k: int | None = None,
        prompt: str | None = None,
    ) -> list[Any]:
        from ..retrieval.in_memory import SearchResult

        scores = self.score(query, documents, prompt=prompt)
        prepared = normalize_inputs(documents)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [
            SearchResult(prepared[int(i)], float(scores[i]), rank)
            for rank, i in enumerate(order, 1)
        ]

    def unload(self) -> None:
        self.model.unload()
