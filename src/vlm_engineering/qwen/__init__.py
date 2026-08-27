# Path: src/vlm_engineering/qwen/__init__.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Qwen3-VL model selection, loading, download and inference."""

from .downloader import download_model_snapshot
from .model import QwenVLModel
from .registry import (
    DEFAULT_QWEN_MODEL_SIZE,
    QWEN3_VL_INSTRUCT_MODELS,
    QwenModelPreset,
    default_model_directory,
    resolve_qwen_model_id,
)

__all__ = [
    "DEFAULT_QWEN_MODEL_SIZE",
    "QWEN3_VL_INSTRUCT_MODELS",
    "QwenModelPreset",
    "QwenVLModel",
    "default_model_directory",
    "download_model_snapshot",
    "resolve_qwen_model_id",
]
