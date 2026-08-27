# Path: src/vlm_engineering/retrieval/reranker.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Qwen3-VL multimodal reranking through SentenceTransformers CrossEncoder."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..config import DEFAULT_QWEN_RERANKER_MODEL
from ..exceptions import OptionalDependencyError


class QwenMultimodalReranker:
    def __init__(
        self,
        model_id: str = DEFAULT_QWEN_RERANKER_MODEL,
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
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise OptionalDependencyError(
                'Install retrieval dependencies with: pip install -e ".[retrieval]"'
            ) from exc
        self._model = CrossEncoder(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )

    def score(self, query: Any, documents: Sequence[Any]) -> np.ndarray:
        self._ensure_loaded()
        pairs = [(query, document) for document in documents]
        return np.asarray(self._model.predict(pairs), dtype=np.float32)
