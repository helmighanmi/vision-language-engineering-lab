# Path: tests/unit/test_providers_qwen.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Tests for Qwen provider wrappers using fakes (no model downloads)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest
from PIL import Image

from vlm_engineering.providers.qwen import (
    QwenEmbeddingProvider,
    QwenRerankerProvider,
    QwenVLMProvider,
)
from vlm_engineering.retrieval.in_memory import SearchResult


class FakeQwenVLModel:
    def __init__(self, **_kwargs: Any) -> None:
        self.calls: list[Any] = []
        self.unloaded = False

    def generate_images(
        self,
        images: Sequence[str | Path],
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 512,
    ) -> str:
        self.calls.append(
            ("images", list(images), prompt, system_prompt, max_new_tokens)
        )
        return "vlm answer"

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        max_new_tokens: int = 512,
    ) -> str:
        self.calls.append(("text", prompt, system_prompt, max_new_tokens))
        return "text answer"

    def unload(self) -> None:
        self.unloaded = True


class FakeQwenEmbedder:
    def __init__(self, **_kwargs: Any) -> None:
        self.calls: list[Any] = []
        self.unloaded = False

    def encode(
        self,
        inputs: Sequence[dict[str, str]],
        *,
        prompt: str | None = None,
    ) -> np.ndarray:
        self.calls.append(("encode", inputs, prompt))
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)

    def embed_image(self, image: str | Path) -> np.ndarray:
        return np.array([0.0, 1.0], dtype=np.float32)

    def unload(self) -> None:
        self.unloaded = True


class FakeQwenReranker:
    def __init__(self, **_kwargs: Any) -> None:
        self.calls: list[Any] = []
        self.unloaded = False

    def score(
        self,
        query: dict[str, str],
        documents: Sequence[dict[str, str]],
        *,
        prompt: str | None = None,
    ) -> np.ndarray:
        self.calls.append(("score", query, documents, prompt))
        return np.array([0.9, 0.1], dtype=np.float32)

    def unload(self) -> None:
        self.unloaded = True


def test_vlm_provider_generate_images() -> None:
    model = FakeQwenVLModel()
    provider = QwenVLMProvider(model=model)  # type: ignore[arg-type]
    answer = provider.generate(["img1.png", "img2.png"], "what is this?", max_new_tokens=100)
    assert answer == "vlm answer"
    assert model.calls[0][0] == "images"
    assert model.calls[0][1] == ["img1.png", "img2.png"]
    assert model.calls[0][2] == "what is this?"
    assert model.calls[0][4] == 100


def test_vlm_provider_generate_text() -> None:
    model = FakeQwenVLModel()
    provider = QwenVLMProvider(model=model)  # type: ignore[arg-type]
    answer = provider.generate_text("only text", max_new_tokens=50)
    assert answer == "text answer"
    assert model.calls[0][0] == "text"
    assert model.calls[0][1] == "only text"
    assert model.calls[0][3] == 50


def test_vlm_provider_unload() -> None:
    model = FakeQwenVLModel()
    provider = QwenVLMProvider(model=model)  # type: ignore[arg-type]
    provider.unload()
    assert model.unloaded


def test_embedding_provider_encode(tmp_path: Path) -> None:
    model = FakeQwenEmbedder()
    provider = QwenEmbeddingProvider(model=model)  # type: ignore[arg-type]
    image = tmp_path / "img.png"
    Image.new("RGB", (8, 8), "blue").save(image)
    vectors = provider.encode(
        [{"text": "a"}, {"image": str(image)}], prompt="query"
    )
    assert vectors.shape == (2, 2)
    assert model.calls[0][0] == "encode"
    assert model.calls[0][2] == "query"


def test_embedding_provider_embed_text_and_image() -> None:
    model = FakeQwenEmbedder()
    provider = QwenEmbeddingProvider(model=model)  # type: ignore[arg-type]
    assert list(provider.embed_text("hello")) == [1.0, 0.0]
    assert list(provider.embed_image("img.png")) == [0.0, 1.0]


def test_embedding_provider_unload() -> None:
    model = FakeQwenEmbedder()
    provider = QwenEmbeddingProvider(model=model)  # type: ignore[arg-type]
    provider.unload()
    assert model.unloaded


def test_reranker_provider_score() -> None:
    model = FakeQwenReranker()
    provider = QwenRerankerProvider(model=model)  # type: ignore[arg-type]
    scores = provider.score("query", [{"text": "a"}, {"text": "b"}])
    assert list(scores) == [0.9, 0.1]
    assert model.calls[0][0] == "score"
    assert model.calls[0][1] == {"text": "query"}


def test_reranker_provider_rerank() -> None:
    model = FakeQwenReranker()
    provider = QwenRerankerProvider(model=model)  # type: ignore[arg-type]
    docs = [{"text": "a"}, {"text": "b"}]
    results = provider.rerank("query", docs, top_k=1)
    assert len(results) == 1
    assert isinstance(results[0], SearchResult)
    assert results[0].rank == 1


def test_reranker_provider_unload() -> None:
    model = FakeQwenReranker()
    provider = QwenRerankerProvider(model=model)  # type: ignore[arg-type]
    provider.unload()
    assert model.unloaded


def test_provider_default_constructs_qwen_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Providers construct real Qwen wrappers when no model is passed."""
    from vlm_engineering.qwen.model import QwenVLModel
    from vlm_engineering.retrieval import QwenVLEmbedder, QwenVLReranker

    monkeypatch.setattr(QwenVLModel, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(QwenVLEmbedder, "__init__", lambda self, **kw: None)
    monkeypatch.setattr(QwenVLReranker, "__init__", lambda self, **kw: None)
    assert QwenVLMProvider().model is not None
    assert QwenEmbeddingProvider().model is not None
    assert QwenRerankerProvider().model is not None
