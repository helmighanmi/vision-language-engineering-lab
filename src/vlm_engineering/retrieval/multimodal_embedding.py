# Path: src/vlm_engineering/retrieval/multimodal_embedding.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Validated Qwen3-VL embeddings through the official Sentence Transformers API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from ..exceptions import InputValidationError
from ..inputs import MultimodalInput, boolean, nonempty_string, normalize_inputs, positive_int
from ..operational import backend_errors
from .backend import RetrievalBackend
from .model_options import EMBEDDING_DIMENSIONS, RetrievalModelOptions, validate_dimensions


class QwenVLEmbedder(RetrievalBackend):
    kind: Literal["Embedding"] = "Embedding"

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
        dimensions: int | None = None,
        normalize_embeddings: bool = True,
        model: Any | None = None,
    ) -> None:
        if model_path is not None and not isinstance(model_path, (str, Path)):
            raise InputValidationError("model_path must be a string or Path.")
        options = RetrievalModelOptions(
            model_size,
            model_id,
            str(model_path) if model_path is not None else None,
            revision,
            device,
            cache_folder,
            trust_remote_code,
            local_files_only,
        )
        positive_int(batch_size, "batch_size")
        boolean(normalize_embeddings, "normalize_embeddings")
        self._native_dimensions = EMBEDDING_DIMENSIONS.get(options.source("Embedding"))
        validate_dimensions(dimensions, self._native_dimensions)
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.normalize_embeddings = normalize_embeddings
        self._configure(options, model)

    def encode(self, inputs: Sequence[MultimodalInput], *, prompt: str | None = None) -> np.ndarray:
        prepared = normalize_inputs(inputs)
        if prompt is not None:
            nonempty_string(prompt, "prompt")
        if not prepared:
            return np.empty((0, self.dimensions or self._native_dimensions or 0), dtype=np.float32)
        model = self._ensure_loaded()
        native = self._native_dimensions
        if native is None:
            native = model.get_sentence_embedding_dimension()
            positive_int(native, "backend embedding dimension")
        validate_dimensions(self.dimensions, native)
        with backend_errors():
            values = model.encode(
                prepared,
                batch_size=self.batch_size,
                prompt=prompt,
                normalize_embeddings=False,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.shape != (len(prepared), native):
            raise InputValidationError(
                f"Unexpected embedding shape {matrix.shape}; expected {(len(prepared), native)}."
            )
        if not np.isfinite(matrix).all():
            raise InputValidationError("Embedding contains NaN/Inf values.")
        matrix = matrix[:, : self.dimensions].copy()
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1, keepdims=True)
        if np.any(norms == 0):
            raise InputValidationError("Embedding contains a zero-norm vector.")
        if self.normalize_embeddings:
            matrix = (matrix / norms).astype(np.float32)
        return matrix

    def embed_text(self, text: str) -> np.ndarray:
        return self.encode([{"text": text}])[0]

    def embed_image(self, image: str | Path) -> np.ndarray:
        return self.encode([{"image": str(image)}])[0]


# Preserve existing import paths and constructor's positional model_id.
QwenMultimodalEmbedder = QwenVLEmbedder
