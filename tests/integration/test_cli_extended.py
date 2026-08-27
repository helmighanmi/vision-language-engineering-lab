# Path: tests/integration/test_cli_extended.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vlm_engineering import cli
from vlm_engineering.documents.schemas import VisualAnalysis


class FakeQwen:
    def generate(self, _image: str, _prompt: str, *, max_new_tokens: int = 512) -> str:
        return f"description:{max_new_tokens}"


def test_cli_help_exits_successfully(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "Vision-Language Engineering Lab CLI" in capsys.readouterr().out


def test_describe_command_uses_qwen_builder(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "_build_qwen", lambda _args: FakeQwen())
    assert cli.main(["describe", "page.png", "--max-new-tokens", "25"]) == 0
    assert capsys.readouterr().out.strip() == "description:25"


def test_analyze_command_outputs_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(cli, "_build_qwen", lambda _args: FakeQwen())
    monkeypatch.setattr(
        cli,
        "analyze_visual_document",
        lambda _model, _image: VisualAnalysis("diagram", "Architecture", "A calls B"),
    )
    assert cli.main(["analyze", "page.png"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["page_type"] == "diagram"
    assert payload["summary"] == "A calls B"


def test_chunk_command_reads_native_text(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    native_text = tmp_path / "native.txt"
    native_text.write_text("Exact component ID: API-42", encoding="utf-8")
    monkeypatch.setattr(cli, "_build_qwen", lambda _args: FakeQwen())
    monkeypatch.setattr(
        cli,
        "analyze_visual_document",
        lambda _model, _image: VisualAnalysis("diagram", "Architecture", "Gateway calls API"),
    )

    exit_code = cli.main(
        [
            "chunk",
            "page.png",
            "--document-id",
            "doc",
            "--source-file",
            "architecture.pdf",
            "--page",
            "3",
            "--native-text-file",
            str(native_text),
            "--parent-context",
            "Backend",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["page"] == 3
    assert payload["parent_context"] == "Backend"
    assert "API-42" in payload["text"]


def test_download_command_prints_destination(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    destination = tmp_path / "model"
    monkeypatch.setattr(cli, "download_model_snapshot", lambda *_args, **_kwargs: destination)
    assert cli.main(["download-model", "--output", str(destination)]) == 0
    assert capsys.readouterr().out.strip() == str(destination)
