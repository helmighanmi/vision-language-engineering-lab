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

from vlm_engineering.exceptions import (
    InputValidationError,
    ModelLoadError,
    OptionalDependencyError,
    VLMEngineeringError,
)

pytest.importorskip("fastapi", reason="the service extra is not installed")

from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from vlm_engineering.service import create_app  # noqa: E402
from vlm_engineering.service.app import _validation_error_response  # noqa: E402
from vlm_engineering.service.config import ServiceConfig  # noqa: E402
from vlm_engineering.service.runtime import ServiceRuntime  # noqa: E402
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


def test_openapi_schema_lists_public_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {"/health", "/ready", "/v1/models"}


def test_unknown_route_is_standard_404(client: TestClient) -> None:
    assert client.get("/v1/not-a-route").status_code == 404


def test_models_catalog_lists_configured_capabilities(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "qwen"
    assert [item["capability"] for item in body["models"]] == [
        "generation",
        "embedding",
        "reranking",
    ]
    assert body["models"][0]["model"].endswith("Qwen3-VL-2B-Instruct")


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


# -- Runtime wiring ----------------------------------------------------------

def test_app_state_exposes_config_and_runtime() -> None:
    app = create_app(ServiceConfig(max_concurrency=3))
    assert app.state.config.max_concurrency == 3
    assert isinstance(app.state.runtime, ServiceRuntime)
    assert app.state.runtime.config is app.state.config
    assert app.state.runtime.loaded_backends() == {
        "generator": False,
        "embedder": False,
        "reranker": False,
    }


def test_create_app_defaults_to_environment_config() -> None:
    app = create_app()
    assert isinstance(app.state.config, ServiceConfig)


# -- Error mapping -----------------------------------------------------------

@pytest.mark.parametrize(
    ("error", "status_code", "code"),
    [
        (InputValidationError("bad input"), 400, "invalid_request"),
        (ValueError("bad value"), 400, "invalid_request"),
        (FileNotFoundError("missing"), 400, "file_not_found"),
        (PermissionError("denied"), 400, "permission_denied"),
        (OptionalDependencyError("install it"), 503, "dependency_unavailable"),
        (ImportError("no module"), 503, "dependency_unavailable"),
        (ModelLoadError("cannot load"), 503, "model_unavailable"),
        (VLMEngineeringError("engine"), 400, "request_error"),
        (RuntimeError("secret-token"), 500, "internal_error"),
    ],
)
def test_exceptions_map_to_error_response(
    error: Exception, status_code: int, code: str
) -> None:
    app = create_app(ServiceConfig())

    @app.get("/boom")
    def boom() -> None:
        raise error

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom")
    assert response.status_code == status_code
    body = response.json()
    assert body["error"] == code
    if status_code == 500:
        assert body["detail"] is None
        assert "secret-token" not in response.text
    else:
        assert body["detail"] == str(error)


def test_lifespan_closes_the_runtime_on_shutdown() -> None:
    class _FakeProvider:
        def __init__(self) -> None:
            self.unload_calls = 0

        def unload(self) -> None:
            self.unload_calls += 1

    fake = _FakeProvider()
    app = create_app(ServiceConfig())
    app.state.runtime = ServiceRuntime(ServiceConfig(), generator=fake)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert fake.unload_calls == 1
