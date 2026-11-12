# Path: tests/unit/test_service_package.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Unit tests for the optional service package surface."""

from __future__ import annotations

import pytest

import vlm_engineering.service as service


def test_dir_includes_lazy_exports() -> None:
    assert set(service.__all__) <= set(dir(service))


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="not_a_real_name"):
        _ = service.not_a_real_name


def test_create_app_is_exported_lazily() -> None:
    pytest.importorskip("fastapi", reason="the service extra is not installed")
    assert service.create_app().state.runtime is not None
