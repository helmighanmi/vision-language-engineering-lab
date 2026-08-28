# Path: tests/unit/test_qwen_downloader.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Unit tests for explicit local Hugging Face model downloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import huggingface_hub
import pytest

from vlm_engineering.qwen.downloader import download_model_snapshot


def test_download_model_snapshot_uses_requested_local_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The downloader must use local_dir instead of the global Hub cache."""
    calls: list[dict[str, Any]] = []

    def fake_snapshot_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(kwargs["local_dir"])

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        fake_snapshot_download,
    )

    destination = tmp_path / "models" / "Qwen3-VL-2B-Instruct"

    result = download_model_snapshot(
        "Qwen/Qwen3-VL-2B-Instruct",
        output_dir=destination,
        revision="main",
    )

    assert result == destination.resolve()
    assert result.is_dir()
    assert calls == [
        {
            "repo_id": "Qwen/Qwen3-VL-2B-Instruct",
            "local_dir": destination.resolve(),
            "revision": "main",
        }
    ]


def test_download_model_snapshot_accepts_existing_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated downloads should be able to reuse/update the same local folder."""
    destination = tmp_path / "model"
    destination.mkdir()

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **_kwargs: str(destination),
    )

    result = download_model_snapshot(
        "Qwen/Qwen3-VL-2B-Instruct",
        output_dir=destination,
    )

    assert result == destination.resolve()


def test_download_model_snapshot_rejects_file_output(
    tmp_path: Path,
) -> None:
    """The model destination must be a directory, not a normal file."""
    destination = tmp_path / "model"
    destination.write_text("not a directory", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Model output path is not a directory",
    ):
        download_model_snapshot(
            "Qwen/Qwen3-VL-2B-Instruct",
            output_dir=destination,
        )


def test_download_model_snapshot_rejects_empty_model_id(
    tmp_path: Path,
) -> None:
    """An empty Hugging Face model ID should fail before any network operation."""
    with pytest.raises(
        ValueError,
        match="model_id must not be empty",
    ):
        download_model_snapshot(
            "   ",
            output_dir=tmp_path / "model",
        )


def test_download_model_snapshot_rejects_empty_output_directory() -> None:
    """An empty output string should not silently resolve to the current directory."""
    with pytest.raises(
        ValueError,
        match="output_dir must not be empty",
    ):
        download_model_snapshot(
            "Qwen/Qwen3-VL-2B-Instruct",
            output_dir="   ",
        )
