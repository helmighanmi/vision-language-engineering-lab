# Path: tests/integration/test_service_api.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Integration tests for the service HTTP API (no model backends needed)."""

from __future__ import annotations

import asyncio
import importlib.metadata

import pytest

pytest.importorskip("fastapi", reason="the service extra is not installed")

from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from vlm_engineering.service import create_app  # noqa: E402
from vlm_engineering.service.app import _validation_error_response  # noqa: E402
from vlm_engineering.service.schemas import package_version  # noqa: E402


@pytest.fixture(name="client")
def client_fixture() -> TestClient:
    return TestClient(create_app())


def test_health_reports_ok_and_package_version(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": package_version(),
    }
    assert package_version() == importlib.metadata.version(
        "vision-language-engineering-lab"
    )


def test_ready_reports_ok(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_openapi_schema_lists_health_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/ready"}


def test_unknown_route_is_standard_404(client: TestClient) -> None:
    assert client.get("/v1/models").status_code == 404


def test_validation_error_response_shape() -> None:
    """The 422 handler maps FastAPI issues onto ErrorResponse fields."""
    exc = RequestValidationError(
        errors=[
            {"loc": ("body", "image"), "msg": "field required"},
            {"loc": ("body", "top_k"), "msg": "must be positive"},
        ],
        body=None,
    )
    response = asyncio.run(_validation_error_response(None, exc))  # type: ignore[arg-type]
    assert response.status_code == 422
    body = response.body  # type: ignore[attr-defined]
    assert b"validation_error" in body
    assert b"body.image: field required" in body
    assert b"body.top_k: must be positive" in body


def test_validation_error_response_with_no_issues() -> None:
    exc = RequestValidationError(errors=[], body=None)
    response = asyncio.run(_validation_error_response(None, exc))  # type: ignore[arg-type]
    assert response.status_code == 422
