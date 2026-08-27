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
from ..exceptions import OptionalDependencyError


class TextEmbedder:
    def __init__(self, model_id: str = DEFAULT_TEXT_EMBEDDER, *, model: Any | None = None) -> None:
        self.model_id = model_id
        self._model = model

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise OptionalDependencyError(
                'Install retrieval dependencies with: pip install -e ".[retrieval]"'
            ) from exc
        self._model = SentenceTransformer(self.model_id)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        values = self._model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(values, dtype=np.float32)
