# Path: tests/unit/test_cli_model_management.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""CLI contract tests for model download and explicit local model loading."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from vlm_engineering import cli


def test_download_model_parser_accepts_2b_preset() -> None:
    """Users should be able to request a managed local 2B download."""
    args = cli.build_parser().parse_args(
        [
            "download-model",
            "--model-size",
            "2b",
            "--output",
            "models/Qwen3-VL-2B-Instruct",
        ]
    )

    assert args.command == "download-model"
    assert args.model_size == "2b"
    assert args.model_id is None
    assert args.output == "models/Qwen3-VL-2B-Instruct"


def test_describe_parser_accepts_explicit_local_model() -> None:
    """Describe should expose an explicit model-path deployment mode."""
    args = cli.build_parser().parse_args(
        [
            "describe",
            "data/diagram_random_clean.png",
            "--model-path",
            "models/Qwen3-VL-2B-Instruct",
        ]
    )

    assert args.command == "describe"
    assert args.model_path == "models/Qwen3-VL-2B-Instruct"
    assert args.model_size is None
    assert args.model_id is None


def test_model_selection_arguments_are_mutually_exclusive() -> None:
    """A command must not silently combine local and Hub model sources."""
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "describe",
                "image.png",
                "--model-size",
                "2b",
                "--model-path",
                "models/Qwen3-VL-2B-Instruct",
            ]
        )


def test_cmd_download_uses_default_project_model_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Without --output, a preset should resolve to models/<model-name>."""
    captured: dict[str, object] = {}

    def fake_download(
        model_id: str,
        *,
        output_dir: str | Path,
        revision: str | None = None,
    ) -> Path:
        captured["model_id"] = model_id
        captured["output_dir"] = output_dir
        captured["revision"] = revision
        return tmp_path / "Qwen3-VL-2B-Instruct"

    monkeypatch.setattr(cli, "download_model_snapshot", fake_download)

    args = Namespace(
        model_size="2b",
        model_id=None,
        output=None,
        revision=None,
    )

    result = cli.cmd_download(args)

    assert result == 0
    assert captured["model_id"] == "Qwen/Qwen3-VL-2B-Instruct"
    assert captured["output_dir"] == Path("models/Qwen3-VL-2B-Instruct")


def test_build_qwen_uses_local_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--model-path should route through QwenVLModel.from_local()."""
    sentinel = object()
    captured: dict[str, object] = {}

    def fake_from_local(
        model_path: str,
        **kwargs: object,
    ) -> object:
        captured["model_path"] = model_path
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        cli.QwenVLModel,
        "from_local",
        fake_from_local,
    )

    args = Namespace(
        model_path="models/Qwen3-VL-2B-Instruct",
        model_id=None,
        model_size=None,
        trust_remote_code=False,
    )

    result = cli._build_qwen(args)

    assert result is sentinel
    assert captured == {
        "model_path": "models/Qwen3-VL-2B-Instruct",
        "trust_remote_code": False,
    }
