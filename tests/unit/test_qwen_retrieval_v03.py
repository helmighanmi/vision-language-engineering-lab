# Path: tests/unit/test_qwen_retrieval_v03.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Boundary and failure tests independent of torch, network and model weights."""

from __future__ import annotations

import errno
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vlm_engineering import QwenVLEmbedder, QwenVLReranker
from vlm_engineering.exceptions import ModelLoadError, OptionalDependencyError, VLMEngineeringError
from vlm_engineering.inputs import normalize_input, normalize_inputs, validate_image
from vlm_engineering.operational import backend_errors
from vlm_engineering.retrieval.model_options import RetrievalModelOptions, validate_snapshot


class EmbeddingBackend:
    def __init__(self, dimension=2048, output=None):
        self.dimension = dimension
        self.output = output
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, inputs, **kwargs):
        self.calls.append((inputs, kwargs))
        if isinstance(self.output, Exception):
            raise self.output
        if self.output is not None:
            return self.output
        return np.tile(np.arange(1, self.dimension + 1, dtype=np.float32), (len(inputs), 1))


class ScoringBackend:
    def __init__(self, output=None):
        self.output = output
        self.calls = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        if isinstance(self.output, Exception):
            raise self.output
        values = np.arange(len(pairs), dtype=np.float32) if self.output is None else np.asarray(self.output)
        return kwargs["activation_fn"](values)


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "diagram with space.png"
    Image.new("RGB", (8, 8), "red").save(path)
    return path


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "snapshot"
    path.mkdir()
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json"):
        (path / name).write_text("{}")
    (path / "model.safetensors").write_bytes(b"structural-test-placeholder-not-weights")
    return path


@pytest.mark.parametrize("cls,kind", [(QwenVLEmbedder, "Embedding"), (QwenVLReranker, "Reranker")])
@pytest.mark.parametrize("size", ["2b", "8b", "8B"])
def test_presets(cls, kind, size):
    model = cls(model_size=size)
    assert model.model_source == f"Qwen/Qwen3-VL-{kind}-{size.upper()}"
    assert model.batch_size == 1
    assert model.trust_remote_code is False


@pytest.mark.parametrize("cls", [QwenVLEmbedder, QwenVLReranker])
def test_custom_and_constructor_options(cls, monkeypatch):
    calls = []
    backend = EmbeddingBackend() if cls is QwenVLEmbedder else ScoringBackend()

    def factory(source, **kwargs):
        calls.append((source, kwargs))
        return backend

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=factory, CrossEncoder=factory),
    )
    model = cls("org/custom", revision="sha", device="cpu", cache_folder="cache", local_files_only=True)
    model._ensure_loaded()
    model._ensure_loaded()
    assert calls == [
        (
            "org/custom",
            {
                "revision": "sha",
                "device": "cpu",
                "cache_folder": "cache",
                "local_files_only": True,
                "trust_remote_code": False,
            },
        )
    ]
    model.unload()
    model._ensure_loaded()
    assert len(calls) == 2


@pytest.mark.parametrize("cls", [QwenVLEmbedder, QwenVLReranker])
def test_local_model_never_falls_back(cls, snapshot, monkeypatch):
    captured = []

    def factory(source, **kwargs):
        captured.append((source, kwargs))
        return object()

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=factory, CrossEncoder=factory),
    )
    model = cls(model_path=snapshot)
    model._ensure_loaded()
    assert captured[0][0] == str(snapshot.resolve())
    assert captured[0][1]["local_files_only"] is True


@pytest.mark.parametrize("cls", [QwenVLEmbedder, QwenVLReranker])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_path": 1},
        {"model_size": "4b"},
        {"model_size": ""},
        {"model_id": ""},
        {"model_size": 2},
        {"model_size": "2b", "model_id": "org/custom"},
        {"model_id": "org/custom", "model_path": "x"},
        {"model_size": "2b", "model_path": "x"},
        {"device": 1},
        {"revision": []},
        {"trust_remote_code": "false"},
        {"local_files_only": 1},
        {"batch_size": 0},
        {"batch_size": -1},
        {"batch_size": True},
        {"batch_size": 1.5},
    ],
)
def test_invalid_model_configuration(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


@pytest.mark.parametrize("cls", [QwenVLEmbedder, QwenVLReranker])
def test_missing_and_non_directory_local_model(cls, tmp_path):
    with pytest.raises(FileNotFoundError):
        cls(model_path=tmp_path / "missing")
    path = tmp_path / "file"
    path.touch()
    with pytest.raises(ValueError):
        cls(model_path=path)
    with pytest.raises(ModelLoadError, match="Incomplete"):
        cls(model_path=tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors",
    ],
)
def test_snapshot_missing_components(snapshot, name):
    (snapshot / name).unlink()
    with pytest.raises(ModelLoadError, match="Incomplete"):
        validate_snapshot(str(snapshot))


@pytest.mark.parametrize(
    "content", ['{"weight_map":{"a":"missing.safetensors"}}', '{"weight_map":{}}', "{}", "[]", "bad json"]
)
def test_snapshot_index_validation(snapshot, content):
    (snapshot / "model.safetensors.index.json").write_text(content)
    with pytest.raises(ModelLoadError):
        validate_snapshot(str(snapshot))


def test_snapshot_valid_shards_and_corrupt_metadata(snapshot):
    (snapshot / "model.safetensors.index.json").write_text('{"weight_map":{"a":"model.safetensors"}}')
    validate_snapshot(str(snapshot))
    (snapshot / "config.json").write_text("[]")
    with pytest.raises(ModelLoadError):
        validate_snapshot(str(snapshot))
    (snapshot / "config.json").write_text("{broken")
    with pytest.raises(ModelLoadError, match="metadata"):
        validate_snapshot(str(snapshot))


def test_zero_size_weight(snapshot):
    (snapshot / "model.safetensors").write_bytes(b"")
    with pytest.raises(ModelLoadError, match="non-empty"):
        validate_snapshot(str(snapshot))


@pytest.mark.parametrize("dimensions", [0, 1, 63, 2049, -1, True, 64.0, "128"])
def test_invalid_embedding_dimensions(dimensions):
    with pytest.raises(ValueError, match="dimensions"):
        QwenVLEmbedder(dimensions=dimensions)


@pytest.mark.parametrize("size,dimension", [("2b", 64), ("2b", 2048), ("8b", 4096)])
def test_dimension_boundaries_and_normalization(size, dimension):
    native = 2048 if size == "2b" else 4096
    backend = EmbeddingBackend(native)
    model = QwenVLEmbedder(model_size=size, dimensions=dimension, model=backend)
    values = model.encode(["one", {"text": "two"}])
    assert values.shape == (2, dimension)
    np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, rtol=1e-6)
    expected = np.arange(1, dimension + 1, dtype=float)
    np.testing.assert_allclose(values[0], expected / np.linalg.norm(expected), rtol=1e-6)
    assert backend.calls[0][1]["batch_size"] == 1


def test_custom_dimension_rejected_before_encode():
    backend = EmbeddingBackend(128)
    with pytest.raises(ValueError, match="dimensions"):
        QwenVLEmbedder("org/custom", dimensions=256, model=backend).embed_text("test")
    assert backend.calls == []


def test_custom_dimensions_raw_and_empty():
    backend = EmbeddingBackend(128)
    model = QwenVLEmbedder("org/custom", dimensions=64, normalize_embeddings=False, model=backend)
    np.testing.assert_array_equal(model.embed_text("test"), np.arange(1, 65))
    assert model.encode([]).shape == (0, 64)
    assert QwenVLEmbedder().encode([]).shape == (0, 2048)


@pytest.mark.parametrize(
    "output",
    [
        np.ones((2, 2048)),
        np.ones((1, 2047)),
        np.ones(2048),
        np.full((1, 2048), np.nan),
        np.full((1, 2048), np.inf),
        np.zeros((1, 2048)),
    ],
)
def test_bad_embedding_backend_output(output):
    with pytest.raises(ValueError):
        QwenVLEmbedder(model=EmbeddingBackend(output=output)).embed_text("test")


def test_zero_truncated_vector():
    values = np.ones((1, 2048))
    values[:, :64] = 0
    with pytest.raises(ValueError, match="zero-norm"):
        QwenVLEmbedder(dimensions=64, model=EmbeddingBackend(output=values)).embed_text("test")


def test_text_image_mixed_and_uri(image_path):
    backend = EmbeddingBackend()
    model = QwenVLEmbedder(model=backend)
    assert model.embed_image(image_path).shape == (2048,)
    result = model.encode(
        [{"text": "diagram"}, {"image": image_path.as_uri()}, {"text": "mixed", "image": str(image_path)}],
        prompt="Find evidence",
    )
    assert result.shape == (3, 2048)
    assert backend.calls[-1][0][1] == {"image": str(image_path)}
    assert backend.calls[-1][0][2] == {"text": "mixed", "image": str(image_path)}
    assert backend.calls[-1][1]["prompt"] == "Find evidence"


@pytest.mark.parametrize(
    "value",
    [
        {},
        {"video": "x"},
        {"text": ""},
        {"text": None},
        {"image": 12},
        {"text": "ok", "unknown": "x"},
        [],
        1,
        " ",
        "missing.png",
        "file:///tmp/x",
    ],
)
def test_strict_input(value):
    with pytest.raises(ValueError):
        normalize_input(value)


@pytest.mark.parametrize("value", ["a string", {"text": "x"}, None, 42])
def test_strict_batch(value):
    with pytest.raises(ValueError):
        normalize_inputs(value)


def test_bad_image_before_loading(tmp_path, monkeypatch):
    model = QwenVLEmbedder()
    monkeypatch.setattr(model, "_ensure_loaded", lambda: pytest.fail("loaded before validation"))
    with pytest.raises(FileNotFoundError):
        model.embed_image(tmp_path / "missing.png")
    with pytest.raises(ValueError, match="not a file"):
        model.embed_image(tmp_path)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="corrupt"):
        model.embed_image(bad)
    gif = tmp_path / "animated.gif"
    Image.new("RGB", (4, 4)).save(gif)
    with pytest.raises(ValueError, match="Unsupported"):
        model.embed_image(gif)


@pytest.mark.parametrize(
    "value", ["http://example.com/x.png", "file://server/x", "data:image/png,x", "file:///x?query=1", ""]
)
def test_unsupported_image_references(value):
    with pytest.raises(ValueError):
        validate_image(value)


def test_reranking_raw_normalized_stable_and_original_items(image_path):
    docs = [{"text": "a"}, {"image": str(image_path)}, {"text": "b", "image": str(image_path)}]
    backend = ScoringBackend([2.0, 2.0, -2.0])
    scorer = QwenVLReranker(model=backend)
    np.testing.assert_array_equal(scorer.score("query", docs), [2, 2, -2])
    ranked = scorer.rerank({"image": str(image_path)}, docs, top_k=2)
    assert ranked[0].item is docs[0]
    assert ranked[1].item is docs[1]
    assert [r.rank for r in ranked] == [1, 2]
    assert backend.calls[-1][1]["batch_size"] == 1
    normalized = QwenVLReranker(model=backend, normalize_scores=True).score("query", docs)
    np.testing.assert_allclose(normalized, 1 / (1 + np.exp(-np.array([2, 2, -2]))), rtol=1e-6)


def test_sigmoid_extreme_logits_and_column_scores():
    result = QwenVLReranker(model=ScoringBackend([[-1000], [1000]]), normalize_scores=True).score(
        "q", ["a", "b"]
    )
    np.testing.assert_array_equal(result, [0, 1])


@pytest.mark.parametrize("output", [[1, 2], [], [[1, 2]], [np.nan], [np.inf]])
def test_bad_reranker_scores(output):
    with pytest.raises(ValueError):
        QwenVLReranker(model=ScoringBackend(output)).score("query", ["one"])


@pytest.mark.parametrize("query,docs", [({}, ["x"]), ("", ["x"]), ("q", [{"image": 12}]), ("q", "text")])
def test_bad_reranker_inputs(query, docs):
    with pytest.raises(ValueError):
        QwenVLReranker().score(query, docs)


def test_reranker_empty_and_invalid_top_k():
    assert QwenVLReranker().score("q", []).shape == (0,)
    assert QwenVLReranker().rerank("q", []) == []
    with pytest.raises(ValueError):
        QwenVLReranker().rerank("q", [], top_k=0)


@pytest.mark.parametrize(
    "exc,fragment",
    [
        (MemoryError(), "Memory"),
        (RuntimeError("CUDA out of memory"), "Memory"),
        (OSError(errno.ENOSPC, "full"), "Disk"),
        (OSError(errno.EDQUOT, "quota"), "quota"),
        (RuntimeError("operator torchvision::nms does not exist"), "torchvision"),
        (RuntimeError("qwen-vl-utils missing"), "qwen-vl-utils"),
        (ValueError("offload whole model to disk"), "offloading"),
        (RuntimeError("safetensor header corrupt"), "corrupt"),
        (OSError("offline cache missing"), "Offline"),
    ],
)
@pytest.mark.parametrize("cls", [QwenVLEmbedder, QwenVLReranker])
def test_operational_backend_errors(cls, exc, fragment):
    backend = EmbeddingBackend(output=exc) if cls is QwenVLEmbedder else ScoringBackend(exc)
    wrapper = cls(model=backend)
    with pytest.raises(VLMEngineeringError, match=fragment) as caught:
        if cls is QwenVLEmbedder:
            wrapper.embed_text("q")
        else:
            wrapper.score("q", ["d"])
    assert caught.value.__cause__ is exc


@pytest.mark.parametrize(
    "name,fragment",
    [
        ("RevisionNotFoundError", "revision"),
        ("LocalEntryNotFoundError", "Offline"),
        ("GatedRepoError", "401/403"),
        ("RepositoryNotFoundError", "model ID"),
        ("HfHubHTTPError", "access"),
        ("SafetensorError", "corrupt"),
    ],
)
def test_hub_error_categories(name, fragment):
    error = type(name, (RuntimeError,), {})("failure")
    with pytest.raises(ModelLoadError, match=fragment):
        with backend_errors(loading=True):
            raise error


@pytest.mark.parametrize("cls", [QwenVLEmbedder, QwenVLReranker])
def test_missing_optional_dependency(cls, monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(OptionalDependencyError, match="qwen-retrieval"):
        cls()._ensure_loaded()


@pytest.mark.parametrize(
    "exc", [RuntimeError("internal bug"), TypeError("bad call"), KeyError("missing field")]
)
def test_programming_errors_preserve_identity(exc):
    with pytest.raises(type(exc)) as caught:
        with backend_errors(loading=True):
            raise exc
    assert caught.value is exc


def test_load_failure_and_none_model(monkeypatch):
    def failure(*args, **kwargs):
        raise OSError("could not load config")

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=failure, CrossEncoder=failure),
    )
    with pytest.raises(ModelLoadError, match="snapshot"):
        QwenVLEmbedder()._ensure_loaded()
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=lambda *a, **k: None, CrossEncoder=None),
    )
    with pytest.raises(ModelLoadError, match="Unable"):
        QwenVLEmbedder()._ensure_loaded()


def test_invalid_booleans_and_prompt():
    with pytest.raises(ValueError):
        QwenVLEmbedder(normalize_embeddings="true")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        QwenVLReranker(normalize_scores=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        QwenVLEmbedder().encode([], prompt="")
    with pytest.raises(ValueError):
        QwenVLReranker().score("q", [], prompt="")
    assert RetrievalModelOptions().source("Reranker") == "Qwen/Qwen3-VL-Reranker-2B"
