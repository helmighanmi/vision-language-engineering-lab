# Path: tests/unit/test_text_embedding_loading.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Deterministic tests for lazy sentence-transformers loading."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from vlm_engineering.exceptions import ModelLoadError, OptionalDependencyError
from vlm_engineering.retrieval.text_embedding import TextEmbedder


class _SentenceModel:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.calls: list[tuple[list[str], bool]] = []

    def encode(self, texts: list[str], *, normalize_embeddings: bool) -> np.ndarray:
        self.calls.append((texts, normalize_embeddings))
        return np.ones((len(texts), 2), dtype=np.float64)


def _install_sentence_transformers(
    monkeypatch: pytest.MonkeyPatch,
    model: object,
) -> list[str]:
    created: list[str] = []

    def factory(model_id: str) -> object:
        created.append(model_id)
        return model

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return created


def test_ensure_loaded_builds_the_model_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _SentenceModel("sentence-transformers/all-MiniLM-L6-v2")
    created = _install_sentence_transformers(monkeypatch, model)

    embedder = TextEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    loaded = embedder._ensure_loaded()

    assert loaded is model
    assert created == ["sentence-transformers/all-MiniLM-L6-v2"]
    assert embedder._ensure_loaded() is model
    assert created == ["sentence-transformers/all-MiniLM-L6-v2"]


def test_ensure_loaded_reports_the_retrieval_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(OptionalDependencyError, match=r"\[retrieval\]"):
        TextEmbedder()._ensure_loaded()


def test_ensure_loaded_raises_when_the_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_sentence_transformers(monkeypatch, None)
    with pytest.raises(ModelLoadError, match="Unable to load text embedder"):
        TextEmbedder("broken/model")._ensure_loaded()


def test_lazy_encode_returns_float32_normalized_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _SentenceModel("sentence-transformers/all-MiniLM-L6-v2")
    _install_sentence_transformers(monkeypatch, model)

    result = TextEmbedder("sentence-transformers/all-MiniLM-L6-v2").encode(["one", "two"])

    assert result.shape == (2, 2)
    assert result.dtype == np.float32
    assert model.calls == [(["one", "two"], True)]
