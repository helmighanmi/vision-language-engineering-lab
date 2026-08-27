# Path: src/vlm_engineering/config.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Central configuration constants and environment-backed defaults."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_QWEN_MODEL = "Qwen/Qwen3-VL-2B-Instruct"
DEFAULT_QWEN_EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_QWEN_RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_TEXT_EMBEDDER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def model_cache_dir() -> Path:
    """Return the configured Hugging Face cache directory."""
    raw = os.getenv("HF_HOME") or os.getenv("VLM_MODEL_CACHE") or ".cache/huggingface"
    return Path(raw).expanduser()
