# Path: tests/unit/test_qwen_model_extended.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vlm_engineering.qwen.model import QwenVLModel


class FakeInputIds:
    shape = (1, 3)


class FakeInputs(dict[str, Any]):
    def to(self, device: str) -> "FakeInputs":
        self["moved_to"] = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] | None = None

    def apply_chat_template(self, messages: list[dict[str, Any]], **_kwargs: Any) -> FakeInputs:
        self.messages = messages
        return FakeInputs({"input_ids": FakeInputIds()})

    def decode(self, generated: Any, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        assert generated == [99, 100]
        return "  grounded answer  "


class FakeModel:
    device = "cpu"

    def __init__(self) -> None:
        self.max_new_tokens: int | None = None

    def generate(self, **kwargs: Any) -> list[list[int]]:
        self.max_new_tokens = int(kwargs["max_new_tokens"])
        return [[10, 11, 12, 99, 100]]


def test_from_hub_uses_online_cache_mode() -> None:
    model = QwenVLModel.from_hub("org/model")
    assert model.model_source == "org/model"
    assert model.local_files_only is False
    assert model.trust_remote_code is False


def test_from_local_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        QwenVLModel.from_local(tmp_path / "missing")


def test_from_local_resolves_directory_and_forces_offline(tmp_path: Path) -> None:
    path = tmp_path / "model"
    path.mkdir()
    model = QwenVLModel.from_local(path)
    assert Path(model.model_source) == path.resolve()
    assert model.local_files_only is True


def test_injected_dependencies_are_reused() -> None:
    fake_model = FakeModel()
    fake_processor = FakeProcessor()
    wrapper = QwenVLModel(model=fake_model, processor=fake_processor)
    model, processor = wrapper._ensure_loaded()
    assert model is fake_model
    assert processor is fake_processor


def test_generate_builds_multimodal_message_and_decodes(tmp_path: Path) -> None:
    image = tmp_path / "page.png"
    image.write_bytes(b"fake")
    fake_model = FakeModel()
    fake_processor = FakeProcessor()
    wrapper = QwenVLModel(model=fake_model, processor=fake_processor)

    answer = wrapper.generate(
        image,
        "Describe the architecture",
        system_prompt="Be factual",
        max_new_tokens=123,
    )

    assert answer == "grounded answer"
    assert fake_model.max_new_tokens == 123
    assert fake_processor.messages is not None
    assert fake_processor.messages[0]["role"] == "system"
    user_content = fake_processor.messages[-1]["content"]
    assert user_content[0]["type"] == "image"
    assert str(user_content[0]["url"]).startswith("file://")
    assert user_content[1]["text"] == "Describe the architecture"
