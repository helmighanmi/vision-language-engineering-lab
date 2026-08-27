# Path: src/vlm_engineering/retrieval/multimodal_embedding.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Qwen3-VL multimodal embedding through SentenceTransformers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..config import DEFAULT_QWEN_EMBEDDING_MODEL
from ..exceptions import OptionalDependencyError


class QwenMultimodalEmbedder:
    """Embed text, image references, or {'text': ..., 'image': ...} dictionaries."""

    def __init__(
        self,
        model_id: str = DEFAULT_QWEN_EMBEDDING_MODEL,
        *,
        trust_remote_code: bool = False,
        model: Any | None = None,
    ) -> None:
        self.model_id = model_id
        self.trust_remote_code = trust_remote_code
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
        self._model = SentenceTransformer(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )

    def encode(self, inputs: Sequence[Any], *, prompt: str | None = None) -> np.ndarray:
        self._ensure_loaded()
        kwargs = {"prompt": prompt} if prompt else {}
        values = self._model.encode(list(inputs), **kwargs)
        return np.asarray(values, dtype=np.float32)
