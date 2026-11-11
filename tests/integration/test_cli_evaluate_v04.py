# Path: tests/integration/test_cli_evaluate_v04.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""CLI retrieval-evaluation integration tests with a deterministic fake embedder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vlm_engineering import cli


class FakeEmbedder:
    def __init__(self, **_kwargs) -> None:
        self.calls = 0

    def encode(self, inputs, *, prompt=None):
        self.calls += 1
        rows = []
        for item in inputs:
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            rows.append([1.0, 0.0] if "cache" in text else [0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)


def _chunk(chunk_id: str, text: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "document_id": "doc",
        "source_file": "source.pdf",
        "page": 1,
        "chunk_type": "diagram",
        "title": "Title",
        "text": text,
        "image_ref": "",
    }


@pytest.fixture
def evaluation_files(tmp_path: Path) -> tuple[Path, Path]:
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(
        "\n".join([json.dumps(_chunk("cache-id", "cache")), json.dumps(_chunk("db-id", "database"))])
        + "\n",
        encoding="utf-8",
    )
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "\n".join(
            [
                json.dumps({"case_id": "c1", "query": "cache", "relevant_ids": ["cache-id"]}),
                json.dumps({"case_id": "c2", "query": "database", "relevant_ids": ["db-id"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return cases, chunks


def test_evaluate_cli_writes_retrieval_report(
    monkeypatch: pytest.MonkeyPatch,
    evaluation_files: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    cases, chunks = evaluation_files
    monkeypatch.setattr(cli, "QwenVLEmbedder", FakeEmbedder)
    output = tmp_path / "report.json"

    exit_code = cli.main(
        [
            "evaluate",
            "--cases",
            str(cases),
            "--chunks",
            str(chunks),
            "--k",
            "1",
            "2",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["aggregate"]["recall@1"] == 1.0
    assert payload["case_count"] == 2

    assert cli.main(["evaluate", "--cases", str(cases), "--chunks", str(chunks), "--k", "1"]) == 0


def test_evaluate_cli_rejects_non_positive_k(
    monkeypatch: pytest.MonkeyPatch,
    evaluation_files: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases, chunks = evaluation_files
    monkeypatch.setattr(cli, "QwenVLEmbedder", FakeEmbedder)
    exit_code = cli.main(
        ["evaluate", "--cases", str(cases), "--chunks", str(chunks), "--k", "0"]
    )
    assert exit_code == 2
    assert "positive integers" in capsys.readouterr().err
