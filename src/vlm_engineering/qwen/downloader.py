# Path: src/vlm_engineering/qwen/downloader.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Explicit Hugging Face model download helpers for offline/local execution."""

from __future__ import annotations

from pathlib import Path

from ..config import DEFAULT_QWEN_MODEL
from ..exceptions import OptionalDependencyError


def download_model_snapshot(
    model_id: str = DEFAULT_QWEN_MODEL,
    *,
    output_dir: str | Path,
    revision: str | None = None,
) -> Path:
    """Download a complete model snapshot to a user-controlled directory."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise OptionalDependencyError(
            'Install Qwen dependencies with: pip install -e ".[qwen]"'
        ) from exc

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=destination, revision=revision)
    return destination
