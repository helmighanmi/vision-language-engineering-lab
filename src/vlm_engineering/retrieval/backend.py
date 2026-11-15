# Path: src/vlm_engineering/retrieval/backend.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Lazy Sentence Transformers loading shared by embedding and reranking."""

from __future__ import annotations

import gc
import threading
from typing import Any

from ..exceptions import ModelLoadError
from ..operational import backend_errors
from .model_options import RetrievalKind, RetrievalModelOptions, validate_snapshot


class RetrievalBackend:
    kind: RetrievalKind

    def __init__(self) -> None:
        self._load_lock = threading.Lock()

    def _configure(self, options: RetrievalModelOptions, model: Any | None) -> None:
        self.options = options
        self.model_id = options.source(self.kind)
        self.model_source = self.model_id
        self.trust_remote_code = options.trust_remote_code
        self.local_files_only = options.local_files_only or options.model_path is not None
        if options.model_path is not None:
            validate_snapshot(self.model_source)
        self._model = model

    def _ensure_loaded(self) -> Any:
        # Fast path: model already loaded.
        if self._model is not None:
            return self._model
        # Slow path: serialize first-load so concurrent callers get one model.
        with self._load_lock:
            if self._model is None:
                with backend_errors(loading=True):
                    from sentence_transformers import CrossEncoder, SentenceTransformer

                    factory = SentenceTransformer if self.kind == "Embedding" else CrossEncoder
                    self._model = factory(self.model_source, **self.options.backend_kwargs())
                if self._model is None:
                    raise ModelLoadError(f"Unable to load {self.model_source}.")
        return self._model

    def unload(self) -> None:
        """Release this wrapper's model reference; external references may retain memory."""
        self._model = None
        gc.collect()
