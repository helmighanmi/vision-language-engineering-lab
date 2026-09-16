# Path: tests/unit/test_runtime_config.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Strict YAML runtime schema and lazy construction tests."""

import sys

import pytest

from vlm_engineering.runtime_config import (
    EmbeddingConfig,
    RAGConfig,
    RerankerConfig,
    RuntimeConfig,
    load_runtime_config,
)


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "[]",
        "{}",
        "version: [",
        "version: 2",
        "version: true",
        "version: '1'",
        "version: 1\nunknown: true",
        "version: 1\nembedding: null",
        "version: 1\nembedding: {oops: 1}",
        "version: 1\nembedding: {model: {oops: 1}}",
        "version: 1\nembedding: {batch_size: false}",
        "version: 1\nembedding: {dimensions: 63}",
        "version: 1\nembedding: {dimensions: 2049}",
        "version: 1\nembedding: {normalize_embeddings: 'true'}",
        "version: 1\nreranker: {normalize_scores: 1}",
        "version: 1\nembedding: {model: {model_size: 2b, model_id: org/name}}",
        "version: 1\nembedding: {model: {model_size: 2b, model_path: /local}}",
        "version: 1\nembedding: {model: {model_id: org/name, model_path: /local}}",
        "version: 1\nreranker: {model: {device: 1}}",
        "version: 1\nrag: {top_k: 4, candidate_k: 3}",
        "version: 1\nrag: {max_new_tokens: -1}",
        "version: 1\nrag: {top_k: 1.5}",
        "version: 1\nrag: {other: 1}",
        "version: 1\nembedding: {model: []}",
        "!!python/object/apply:os.system ['false']",
    ],
)
def test_invalid_yaml(tmp_path, payload):
    path = tmp_path / "config.yaml"
    path.write_text(payload)
    with pytest.raises(ValueError):
        load_runtime_config(path)


def test_valid_profile_without_backends(tmp_path, monkeypatch):
    for module in ("sentence_transformers", "torch", "transformers"):
        monkeypatch.setitem(sys.modules, module, None)
    path = tmp_path / "profile.yaml"
    path.write_text("version: 1\nembedding: {model: {model_path: /not/downloaded/yet}, dimensions: 128}")
    config = load_runtime_config(path)
    assert config.embedding.model.model_path == "/not/downloaded/yet"
    assert config.embedding.dimensions == 128
    assert config.rag.candidate_k == 12


def test_example_profile_and_factory():
    config = load_runtime_config("configs/qwen_multimodal_rag.example.yaml")
    pipe = config.build_pipeline(object())
    assert pipe.embedder.dimensions == 1024
    assert pipe.reranker.batch_size == 1
    assert pipe.top_k == 3
    assert pipe.embedder._model is None
    assert pipe.reranker._model is None


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (RuntimeConfig, {"version": 2}),
        (RuntimeConfig, {"embedding": {}}),
        (EmbeddingConfig, {"model": {}}),
        (RerankerConfig, {"model": {}}),
        (RAGConfig, {"candidate_k": 1}),
    ],
)
def test_dataclasses_validate_direct_construction(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


def test_missing_yaml_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "yaml", None)
    from vlm_engineering.exceptions import OptionalDependencyError

    with pytest.raises(OptionalDependencyError, match="config"):
        load_runtime_config("unused.yaml")
