# Path: tests/unit/test_clip_encoder_extended.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from vlm_engineering.clip.encoder import CLIPEncoder


class FakeTensor:
    def __init__(self, value: Any) -> None:
        self._value = np.asarray(value, dtype=np.float32)

    def to(self, _device: str) -> "FakeTensor":
        return self

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class FakeProcessor:
    def __call__(self, **kwargs: Any) -> dict[str, FakeTensor]:
        size = len(kwargs.get("text", kwargs.get("images", [])))
        return {"input": FakeTensor(np.ones((size, 1), dtype=np.float32))}


class FakeClipModel:
    def get_text_features(self, **_kwargs: Any) -> FakeTensor:
        return FakeTensor([[3.0, 4.0], [0.0, 5.0]])

    def get_image_features(self, **_kwargs: Any) -> FakeTensor:
        return FakeTensor([[4.0, 3.0]])


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch = SimpleNamespace(inference_mode=lambda: contextlib.nullcontext())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


def test_normalize_produces_unit_vectors() -> None:
    values = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    normalized = CLIPEncoder._normalize(values)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)


def test_normalize_handles_zero_vector_without_nan() -> None:
    normalized = CLIPEncoder._normalize(np.zeros((1, 3), dtype=np.float32))
    assert np.isfinite(normalized).all()
    assert np.array_equal(normalized, np.zeros((1, 3), dtype=np.float32))


def test_encode_text_with_injected_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch)
    encoder = CLIPEncoder(model=FakeClipModel(), processor=FakeProcessor())
    values = encoder.encode_text(["cat", "dog"])
    assert values.shape == (2, 2)
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0)


def test_encode_images_with_injected_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch(monkeypatch)
    encoder = CLIPEncoder(model=FakeClipModel(), processor=FakeProcessor())
    values = encoder.encode_images([object()])
    assert values.shape == (1, 2)
    assert np.allclose(values[0], np.array([0.8, 0.6], dtype=np.float32))


def test_similarity_returns_text_by_image_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = CLIPEncoder()
    monkeypatch.setattr(
        encoder,
        "encode_text",
        lambda _texts: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        encoder,
        "encode_images",
        lambda _images: np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
    )
    result = encoder.similarity(["a", "b"], [object(), object()])
    assert result.shape == (2, 2)
    assert result[0, 0] == pytest.approx(1.0)


def test_zero_shot_predictions_are_sorted_and_sum_to_one(monkeypatch: pytest.MonkeyPatch) -> None:
    encoder = CLIPEncoder()
    monkeypatch.setattr(
        encoder,
        "similarity",
        lambda _labels, _images: np.array([[0.1], [2.0], [0.5]], dtype=np.float32),
    )
    result = encoder.zero_shot_classify(object(), ["cat", "dog", "bird"])
    assert [item.label for item in result] == ["dog", "bird", "cat"]
    assert sum(item.probability for item in result) == pytest.approx(1.0)
