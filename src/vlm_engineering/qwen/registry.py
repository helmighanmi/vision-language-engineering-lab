# Path: src/vlm_engineering/qwen/registry.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Qwen3-VL model presets and model-selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, slots=True)
class QwenModelPreset:
    """Metadata for a supported Qwen3-VL Instruct preset."""

    size: str
    model_id: str
    parameter_class: str
    description: str


DEFAULT_QWEN_MODEL_SIZE = "2b"

_PRESETS = {
    "2b": QwenModelPreset(
        size="2b",
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        parameter_class="~2B",
        description="Default: lowest resource requirement; recommended for learning and local prototypes.",
    ),
    "4b": QwenModelPreset(
        size="4b",
        model_id="Qwen/Qwen3-VL-4B-Instruct",
        parameter_class="~4B",
        description="Balanced option when you want stronger quality and have more compute/memory.",
    ),
    "8b": QwenModelPreset(
        size="8b",
        model_id="Qwen/Qwen3-VL-8B-Instruct",
        parameter_class="~8B",
        description="Higher-capacity option for stronger inference on more capable hardware.",
    ),
}

QWEN3_VL_INSTRUCT_MODELS: Mapping[str, QwenModelPreset] = MappingProxyType(_PRESETS)


def normalize_model_size(model_size: str) -> str:
    """Normalize and validate a friendly Qwen model-size alias."""
    normalized = model_size.strip().lower()
    if normalized not in QWEN3_VL_INSTRUCT_MODELS:
        supported = ", ".join(QWEN3_VL_INSTRUCT_MODELS)
        raise ValueError(f"Unsupported Qwen model size {model_size!r}. Choose one of: {supported}.")
    return normalized


def resolve_qwen_model_id(
    *,
    model_size: str | None = None,
    model_id: str | None = None,
) -> str:
    """Resolve a preset alias or explicit Hugging Face model ID.

    ``model_size`` and ``model_id`` are intentionally mutually exclusive to
    avoid silently running a different model than the caller intended.
    """
    if model_size is not None and model_id is not None:
        raise ValueError("Provide either model_size or model_id, not both.")
    if model_id is not None:
        value = model_id.strip()
        if not value:
            raise ValueError("model_id must not be empty.")
        return value

    size = normalize_model_size(model_size or DEFAULT_QWEN_MODEL_SIZE)
    return QWEN3_VL_INSTRUCT_MODELS[size].model_id


def default_model_directory(model_id: str) -> Path:
    """Return a deterministic local directory for an explicitly downloaded model."""
    safe_name = model_id.rstrip("/").split("/")[-1]
    return Path("models") / safe_name
