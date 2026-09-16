# Path: tests/contract/test_retrieval_release.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Public API, dependency, opt-in CI and Docker release contracts."""

import tomllib
from pathlib import Path

from vlm_engineering import MultimodalRAGPipeline, QwenVLEmbedder, QwenVLReranker
from vlm_engineering.retrieval import QwenMultimodalEmbedder, QwenMultimodalReranker

ROOT = Path(__file__).resolve().parents[2]


def test_old_and_new_names_share_implementation():
    assert QwenVLEmbedder is QwenMultimodalEmbedder
    assert QwenVLReranker is QwenMultimodalReranker
    assert MultimodalRAGPipeline.__name__ == "MultimodalRAGPipeline"


def test_dependency_contract():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert {"qwen", "retrieval", "qwen-retrieval", "config", "all", "dev"} <= extras.keys()
    for group in ("qwen-retrieval", "all"):
        assert "sentence-transformers[image]>=5.4,<7" in extras[group]
        assert "transformers>=5.0,<6" in extras[group]
    for group in ("qwen", "qwen-retrieval", "all"):
        assert "torchvision>=0.23,<1" in extras[group]
        assert "torch>=2.8,<3" in extras[group]
    assert "PyYAML>=6.0.2,<7" in extras["config"]
    assert data["tool"]["coverage"]["report"]["fail_under"] == 80


def test_normal_ci_excludes_real_models():
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    assert '-m "not real_model"' in workflow
    assert "QwenVLEmbedder" in workflow
    assert "QwenVLReranker" in workflow
    assert "MultimodalRAGPipeline" in workflow
    for version in ("3.11", "3.12", "3.13"):
        assert version in workflow


def test_docker_excludes_local_weights():
    assert "models/" in (ROOT / ".dockerignore").read_text().splitlines()
    assert "models/" in (ROOT / ".gitignore").read_text().splitlines()
    assert "qwen-retrieval" in (ROOT / "Dockerfile").read_text()
