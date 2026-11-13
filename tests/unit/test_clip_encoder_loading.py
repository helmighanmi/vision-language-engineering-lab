# Path: tests/unit/test_clip_encoder_loading.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Deterministic tests for CLIP lazy loading and device placement."""

from __future__ import annotations

import contextlib
import sys
import types
from typing import Any

import numpy as np
import pytest

from vlm_engineering.clip.encoder import CLIPEncoder
from vlm_engineering.exceptions import ModelLoadError, OptionalDependencyError


class _Tensor:
    def __init__(self, value: Any) -> None:
        self._value = np.asarray(value, dtype=np.float32)
        self.devices: list[str] = []

    def to(self, device: str) -> "_Tensor":
        self.devices.append(device)
        return self

    def detach(self) -> "_Tensor":
        return self

    def cpu(self) -> "_Tensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._value


class _Processor:
    def __init__(self) -> None:
        self.tensors: list[_Tensor] = []

    def __call__(self, **kwargs: Any) -> dict[str, _Tensor]:
        size = len(kwargs.get("text", kwargs.get("images", [])))
        tensor = _Tensor(np.ones((size, 2), dtype=np.float32))
        self.tensors.append(tensor)
        return {"input": tensor}


class _Model:
    def __init__(self) -> None:
        self.device: str | None = None

    def to(self, device: str) -> "_Model":
        self.device = device
        return self

    def get_text_features(self, **_kwargs: Any) -> _Tensor:
        return _Tensor([[3.0, 4.0], [0.0, 5.0]])

    def get_image_features(self, **_kwargs: Any) -> _Tensor:
        return _Tensor([[4.0, 3.0]])


class _ModelFactory:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.model_ids: list[str] = []

    def from_pretrained(self, model_id: str) -> Any:
        self.model_ids.append(model_id)
        return self.result


def _fake_torch(*, cuda: bool) -> types.ModuleType:
    module = types.ModuleType("torch")
    module.cuda = types.SimpleNamespace(is_available=lambda: cuda)  # type: ignore[attr-defined]
    module.inference_mode = lambda: contextlib.nullcontext()  # type: ignore[attr-defined]
    return module


def _fake_transformers(model: Any, processor: Any) -> types.ModuleType:
    module = types.ModuleType("transformers")
    module.CLIPModel = model  # type: ignore[attr-defined]
    module.CLIPProcessor = processor  # type: ignore[attr-defined]
    return module


def test_ensure_loaded_builds_model_and_processor_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    processor = _Processor()
    model_factory = _ModelFactory(model)
    processor_factory = _ModelFactory(processor)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_transformers(model_factory, processor_factory),
    )

    encoder = CLIPEncoder("openai/clip-vit-base-patch32")
    loaded_model, loaded_processor = encoder._ensure_loaded()

    assert loaded_model is model
    assert loaded_processor is processor
    assert encoder.device == "cuda"
    assert model.device == "cuda"
    assert model_factory.model_ids == ["openai/clip-vit-base-patch32"]
    assert processor_factory.model_ids == ["openai/clip-vit-base-patch32"]

    encoder._ensure_loaded()
    assert model_factory.model_ids == ["openai/clip-vit-base-patch32"]


def test_ensure_loaded_honours_an_explicit_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_transformers(_ModelFactory(_Model()), _ModelFactory(_Processor())),
    )

    encoder = CLIPEncoder(device="mps")
    model, _ = encoder._ensure_loaded()

    assert encoder.device == "mps"
    assert model.device == "mps"


def test_ensure_loaded_reports_the_clip_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(OptionalDependencyError, match=r"\[clip\]"):
        CLIPEncoder()._ensure_loaded()


def test_ensure_loaded_raises_when_the_processor_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        _fake_transformers(_ModelFactory(_Model()), _ModelFactory(None)),
    )

    with pytest.raises(ModelLoadError, match="Unable to load CLIP model"):
        CLIPEncoder("broken/model")._ensure_loaded()


def test_encode_text_moves_inputs_to_the_configured_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    processor = _Processor()
    encoder = CLIPEncoder(model=_Model(), processor=processor, device="cuda")

    values = encoder.encode_text(["cat", "dog"])

    assert values.shape == (2, 2)
    assert processor.tensors[0].devices == ["cuda"]


def test_encode_images_moves_inputs_to_the_configured_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=False))
    processor = _Processor()
    encoder = CLIPEncoder(model=_Model(), processor=processor, device="cuda")

    values = encoder.encode_images([object()])

    assert values.shape == (1, 2)
    assert processor.tensors[0].devices == ["cuda"]
