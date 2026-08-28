# Path: src/vlm_engineering/qwen/downloader.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Explicit Hugging Face model downloads for predictable local/offline use."""

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
    """Download a complete model snapshot into a user-controlled directory.

    Using Hugging Face ``snapshot_download(..., local_dir=...)`` keeps the model
    files under the requested directory instead of the normal global Hub cache.
    Hugging Face may create a small ``.cache/huggingface`` metadata directory
    inside the destination so repeated downloads can be updated efficiently.

    Authentication is intentionally not accepted as a CLI argument by this
    package. Users can authenticate with ``hf auth login`` or ``HF_TOKEN`` so a
    secret does not have to be placed in shell history.
    """
    normalized_model_id = model_id.strip()
    if not normalized_model_id:
        raise ValueError("model_id must not be empty.")

    if isinstance(output_dir, str) and not output_dir.strip():
        raise ValueError("output_dir must not be empty.")

    destination = Path(output_dir).expanduser().resolve()

    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Model output path is not a directory: {destination}")

    destination.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise OptionalDependencyError(
            'Install Qwen dependencies with: pip install "vision-language-engineering-lab[qwen]"'
        ) from exc

    snapshot_download(
        repo_id=normalized_model_id,
        local_dir=destination,
        revision=revision,
    )

    return destination
