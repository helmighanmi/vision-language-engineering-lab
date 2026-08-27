# Path: tests/unit/test_qwen_registry.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Tests for Qwen3-VL model presets and selection rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_engineering.qwen import (
    DEFAULT_QWEN_MODEL_SIZE,
    QWEN3_VL_INSTRUCT_MODELS,
    QwenVLModel,
    default_model_directory,
    resolve_qwen_model_id,
)


def test_default_model_is_2b_instruct() -> None:
    assert DEFAULT_QWEN_MODEL_SIZE == "2b"
    assert resolve_qwen_model_id() == "Qwen/Qwen3-VL-2B-Instruct"
    assert QwenVLModel().model_source == "Qwen/Qwen3-VL-2B-Instruct"


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ("2b", "Qwen/Qwen3-VL-2B-Instruct"),
        ("4B", "Qwen/Qwen3-VL-4B-Instruct"),
        ("8b", "Qwen/Qwen3-VL-8B-Instruct"),
    ],
)
def test_model_size_presets_resolve_case_insensitively(size: str, expected: str) -> None:
    assert resolve_qwen_model_id(model_size=size) == expected
    assert QwenVLModel(model_size=size).model_source == expected


def test_registry_contains_documented_presets() -> None:
    assert set(QWEN3_VL_INSTRUCT_MODELS) == {"2b", "4b", "8b"}
    assert QWEN3_VL_INSTRUCT_MODELS["4b"].parameter_class == "~4B"


def test_custom_model_id_is_preserved() -> None:
    model_id = "Qwen/custom-compatible-vlm"
    assert resolve_qwen_model_id(model_id=model_id) == model_id
    assert QwenVLModel.from_hub(model_id).model_source == model_id


def test_size_and_model_id_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="either model_size or model_id"):
        resolve_qwen_model_id(model_size="4b", model_id="org/model")


def test_invalid_model_size_has_clear_error() -> None:
    with pytest.raises(ValueError, match="Unsupported Qwen model size"):
        resolve_qwen_model_id(model_size="70b")


def test_constructor_rejects_model_source_and_model_size_together() -> None:
    with pytest.raises(ValueError, match="either model_source or model_size"):
        QwenVLModel("org/model", model_size="4b")


def test_default_download_directory_uses_model_name() -> None:
    assert default_model_directory("Qwen/Qwen3-VL-4B-Instruct") == Path(
        "models/Qwen3-VL-4B-Instruct"
    )
