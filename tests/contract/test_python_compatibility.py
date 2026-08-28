# Path: tests/contract/test_python_compatibility.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Contract tests for supported Python versions and package metadata."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"

SUPPORTED_PYTHON_VERSIONS = {
    "3.11",
    "3.12",
    "3.13",
}


def _load_pyproject() -> dict[str, Any]:
    """Load the repository pyproject configuration."""
    with PYPROJECT.open("rb") as file:
        return tomllib.load(file)


def test_current_python_is_supported() -> None:
    """CI must execute tests only on officially supported Python versions."""
    current_version = f"{sys.version_info.major}.{sys.version_info.minor}"

    assert current_version in SUPPORTED_PYTHON_VERSIONS, (
        f"Python {current_version} is not declared as supported. "
        f"Supported versions: {sorted(SUPPORTED_PYTHON_VERSIONS)}"
    )


def test_requires_python_declares_supported_range() -> None:
    """Package metadata must expose the official compatibility range."""
    pyproject = _load_pyproject()

    requires_python = pyproject["project"]["requires-python"]

    assert requires_python == ">=3.11,<3.14"


def test_python_classifiers_match_supported_versions() -> None:
    """PyPI classifiers must advertise every supported Python version."""
    pyproject = _load_pyproject()

    classifiers = set(pyproject["project"]["classifiers"])

    expected_classifiers = {
        f"Programming Language :: Python :: {version}"
        for version in SUPPORTED_PYTHON_VERSIONS
    }

    assert expected_classifiers <= classifiers


def test_package_version_is_0_2_0() -> None:
    """The compatibility release must carry the expected package version."""
    pyproject = _load_pyproject()

    assert pyproject["project"]["version"] == "0.2.0"


def test_sentence_transformers_support_range_is_declared() -> None:
    """Retrieval extras must allow the validated Sentence Transformers range."""
    pyproject = _load_pyproject()

    retrieval_dependencies = pyproject["project"]["optional-dependencies"]["retrieval"]

    assert "sentence-transformers>=5.4,<7" in retrieval_dependencies


def test_all_extra_contains_sentence_transformers() -> None:
    """The complete runtime extra must include retrieval dependencies."""
    pyproject = _load_pyproject()

    all_dependencies = pyproject["project"]["optional-dependencies"]["all"]

    assert "sentence-transformers>=5.4,<7" in all_dependencies