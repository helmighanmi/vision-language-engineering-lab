# Path: src/vlm_engineering/retrieval/text_embedding.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Text-only embedding adapter for the VLM-description-to-text-RAG pattern."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..config import DEFAULT_TEXT_EMBEDDER
from ..exceptions import ModelLoadError, OptionalDependencyError


class TextEmbedder:
    def __init__(self, model_id: str = DEFAULT_TEXT_EMBEDDER, *, model: Any | None = None) -> None:
        self.model_id = model_id
        self._model = model

    def _ensure_loaded(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise OptionalDependencyError(
                'Install retrieval dependencies with: pip install -e ".[retrieval]"'
            ) from exc

        self._model = SentenceTransformer(self.model_id)
        if self._model is None:  # defensive guard
            raise ModelLoadError(f"Unable to load text embedder {self.model_id!r}.")
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        model = self._ensure_loaded()
        values = model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(values, dtype=np.float32)
