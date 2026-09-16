# Path: src/vlm_engineering/retrieval/reranker.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Validated relevance scoring with explicit raw-logit and sigmoid semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from ..exceptions import InputValidationError
from ..inputs import (
    MultimodalInput,
    boolean,
    nonempty_string,
    normalize_input,
    normalize_inputs,
    positive_int,
)
from ..operational import backend_errors
from .backend import RetrievalBackend
from .in_memory import SearchResult
from .model_options import RetrievalModelOptions


def _identity(values: Any) -> Any:
    return values


class QwenVLReranker(RetrievalBackend):
    kind: Literal["Reranker"] = "Reranker"

    def __init__(
        self,
        model_id: str | None = None,
        *,
        model_size: str | None = None,
        model_path: str | Path | None = None,
        revision: str | None = None,
        device: str | None = None,
        cache_folder: str | None = None,
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        batch_size: int = 1,
        normalize_scores: bool = False,
        model: Any | None = None,
    ) -> None:
        if model_path is not None and not isinstance(model_path, (str, Path)):
            raise InputValidationError("model_path must be a string or Path.")
        positive_int(batch_size, "batch_size")
        boolean(normalize_scores, "normalize_scores")
        self.batch_size = batch_size
        self.normalize_scores = normalize_scores
        self._configure(
            RetrievalModelOptions(
                model_size,
                model_id,
                str(model_path) if model_path is not None else None,
                revision,
                device,
                cache_folder,
                trust_remote_code,
                local_files_only,
            ),
            model,
        )

    def score(
        self, query: MultimodalInput, documents: Sequence[MultimodalInput], *, prompt: str | None = None
    ) -> np.ndarray:
        question = normalize_input(query)
        candidates = normalize_inputs(documents)
        if prompt is not None:
            nonempty_string(prompt, "prompt")
        if not candidates:
            return np.empty(0, dtype=np.float32)
        model = self._ensure_loaded()
        with backend_errors():
            values = model.predict(
                [(question, item) for item in candidates],
                batch_size=self.batch_size,
                activation_fn=_identity,
                convert_to_numpy=True,
                show_progress_bar=False,
                prompt=prompt,
            )
        scores = np.asarray(values, dtype=np.float32)
        if scores.shape == (len(candidates), 1):
            scores = scores[:, 0]
        if scores.shape != (len(candidates),):
            raise InputValidationError(
                "Unexpected reranker score shape/count; expected one score per candidate."
            )
        if not np.isfinite(scores).all():
            raise InputValidationError("Reranker scores contain NaN/Inf values.")
        if self.normalize_scores:
            # Stable sigmoid, exactly once (backend activation is explicitly identity).
            positive = scores >= 0
            output = np.empty_like(scores)
            output[positive] = 1 / (1 + np.exp(-scores[positive]))
            exp = np.exp(scores[~positive])
            output[~positive] = exp / (1 + exp)
            scores = output
        return scores

    def rerank(
        self,
        query: MultimodalInput,
        documents: Sequence[MultimodalInput],
        *,
        top_k: int | None = None,
        prompt: str | None = None,
    ) -> list[SearchResult]:
        if top_k is not None:
            positive_int(top_k, "top_k")
        scores = self.score(query, documents, prompt=prompt)
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [SearchResult(documents[int(i)], float(scores[i]), rank) for rank, i in enumerate(order, 1)]


QwenMultimodalReranker = QwenVLReranker
