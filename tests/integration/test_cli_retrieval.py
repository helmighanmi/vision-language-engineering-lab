# Path: tests/integration/test_cli_retrieval.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Exercise actual CLI routing with fake backend factories and real local inputs."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vlm_engineering import cli


class Model:
    def __init__(self, source, **kwargs):
        self.source = source
        self.kwargs = kwargs

    def encode(self, inputs, **kwargs):
        assert kwargs["batch_size"] == 1
        return np.ones((len(inputs), 2048))

    def predict(self, pairs, **kwargs):
        return np.arange(len(pairs), dtype=np.float32)


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model, CrossEncoder=Model)
    )


def test_embed_text_and_image(backend, tmp_path, capsys):
    image = tmp_path / "test.png"
    Image.new("RGB", (4, 4)).save(image)
    assert cli.main(["embed", "--text", "diagram", "--image", str(image), "--dimensions", "64"]) == 0
    values = np.asarray(json.loads(capsys.readouterr().out))
    assert values.shape == (1, 64)
    assert np.linalg.norm(values) == pytest.approx(1)


def test_rerank_json_original_items(backend, tmp_path, capsys):
    path = tmp_path / "docs.json"
    path.write_text('[{"text": "first"}, {"text": "second"}]')
    assert (
        cli.main(
            [
                "rerank",
                "--query",
                "query",
                "--documents-json",
                str(path),
                "--top-k",
                "1",
                "--normalize-scores",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result[0]["item"] == {"text": "second"}
    assert result[0]["score"] == pytest.approx(0.7310586)
    assert result[0]["rank"] == 1


def test_validate_config_without_model_loading(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert cli.main(["validate-config", "configs/qwen_multimodal_rag.example.yaml"]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == 1


@pytest.mark.parametrize(
    "args,fragment",
    [
        (["embed"], "cannot be empty"),
        (["embed", "--text", ""], "non-empty"),
        (["embed", "--image", "missing.png"], "not found"),
        (["embed", "--text", "q", "--dimensions", "4"], "dimensions"),
        (["embed", "--text", "q", "--batch-size", "0"], "batch_size"),
        (["validate-config", "missing.yaml"], "missing.yaml"),
    ],
)
def test_useful_cli_validation_errors(args, fragment, capsys):
    assert cli.main(args) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert fragment in output.err


def test_invalid_candidate_json(tmp_path, capsys):
    path = tmp_path / "bad.json"
    path.write_text("not json")
    assert cli.main(["rerank", "--query", "q", "--documents-json", str(path)]) == 2
    assert "vlm-lab:" in capsys.readouterr().err


def test_oom_and_unknown_programming_error(monkeypatch, capsys):
    def oom(*args, **kwargs):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(cli, "cmd_embed", oom)
    assert cli.main(["embed", "--text", "q"]) == 2
    assert "Memory exhausted" in capsys.readouterr().err

    def bug(*args, **kwargs):
        raise TypeError("unexpected bug")

    monkeypatch.setattr(cli, "cmd_embed", bug)
    with pytest.raises(TypeError, match="unexpected bug"):
        cli.main(["embed", "--text", "q"])


@pytest.mark.parametrize("flag,kind", [("--embedding-size", "Embedding"), ("--reranker-size", "Reranker")])
def test_download_retrieval_preset(flag, kind, monkeypatch, capsys):
    calls = []

    def download(model_id, **kwargs):
        calls.append((model_id, kwargs))
        return kwargs["output_dir"]

    monkeypatch.setattr(cli, "download_model_snapshot", download)
    assert cli.main(["download-model", flag, "8b", "--revision", "sha"]) == 0
    assert calls[0][0] == f"Qwen/Qwen3-VL-{kind}-8B"
    assert calls[0][1]["revision"] == "sha"
    assert capsys.readouterr().out.strip() == f"models/Qwen3-VL-{kind}-8B"


@pytest.mark.parametrize(
    "args",
    [
        ["embed", "--model-size", "2b", "--model-path", "x"],
        ["rerank", "--query", "q", "--documents-json", "x", "--model-size", "4b"],
        ["download-model", "--embedding-size", "2b", "--reranker-size", "2b"],
    ],
)
def test_parser_selection_conflicts(args):
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2


def test_unexpected_backend_value_error_propagates(monkeypatch):
    def bug(*args, **kwargs):
        raise ValueError("internal tensor algorithm error")

    monkeypatch.setattr(cli, "cmd_embed", bug)
    with pytest.raises(ValueError, match="internal tensor"):
        cli.main(["embed", "--text", "q"])
