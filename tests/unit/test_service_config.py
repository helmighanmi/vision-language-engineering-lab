# Path: tests/unit/test_service_config.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Unit tests for the environment-driven service configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from vlm_engineering.exceptions import InputValidationError
from vlm_engineering.service.config import (
    ALLOW_REMOTE_IMAGES_ENV,
    MAX_BODY_BYTES_ENV,
    MAX_CONCURRENCY_ENV,
    MODEL_PATH_ENV,
    MODEL_SIZE_ENV,
    RATE_LIMIT_ENV,
    REQUEST_TIMEOUT_ENV,
    VLM_BASE_URL_ENV,
    VLM_PROVIDER_ENV,
    RateLimit,
    ServiceConfig,
)


def test_defaults_when_environment_is_empty() -> None:
    config = ServiceConfig.from_env({})
    assert config.api_key is None
    assert config.media_root is None
    assert config.output_root is None
    assert config.allow_remote_images is False
    assert config.max_concurrency == 2
    assert config.public_metrics is True
    assert config.rate_limit == RateLimit(120, 60)
    assert config.max_body_bytes == 10 * 1024 * 1024
    assert config.runtime_config_path is None
    assert config.vlm_provider == "qwen"
    assert config.vlm_base_url is None
    assert config.vlm_model is None
    assert config.model_size is None
    assert config.model_id is None
    assert config.model_path is None
    assert config.request_timeout == 120.0
    assert config.store == "memory"
    assert config.store_path == "index"
    assert config.qdrant_host == "localhost"
    assert config.qdrant_port == 6333
    assert config.qdrant_collection == "vlm_engineering"
    assert config.qdrant_api_key is None
    assert config.qdrant_https is False


def test_from_env_reads_every_supported_variable() -> None:
    config = ServiceConfig.from_env(
        {
            "VLM_SERVICE_API_KEY": "  gateway-secret  ",
            "VLM_SERVICE_MEDIA_ROOT": "~/service-media",
            "VLM_SERVICE_OUTPUT_ROOT": "/tmp/service-output",
            "VLM_SERVICE_ALLOW_REMOTE_IMAGES": "yes",
            "VLM_SERVICE_MAX_CONCURRENCY": "4",
            "VLM_SERVICE_PUBLIC_METRICS": "off",
            "VLM_SERVICE_RATE_LIMIT": "30/hour",
            "VLM_SERVICE_MAX_BODY_BYTES": "2048",
            "VLM_SERVICE_RUNTIME_CONFIG": "/tmp/runtime.yaml",
            "VLM_SERVICE_VLM_PROVIDER": "OPENAI-COMPATIBLE",
            "VLM_SERVICE_VLM_BASE_URL": "https://vllm.internal:8000",
            "VLM_SERVICE_VLM_MODEL": "Qwen/Qwen3-VL-8B-Instruct",
            "VLM_SERVICE_VLM_API_KEY": "provider-secret",
            "VLM_SERVICE_MODEL_SIZE": "8b",
            "VLM_SERVICE_REQUEST_TIMEOUT": "12.5",
            "VLM_SERVICE_STORE": "SQLITE-VEC",
            "VLM_SERVICE_STORE_PATH": "indexes/main",
            "VLM_SERVICE_QDRANT_HOST": "qdrant.internal",
            "VLM_SERVICE_QDRANT_PORT": "6334",
            "VLM_SERVICE_QDRANT_COLLECTION": "documents",
            "VLM_SERVICE_QDRANT_API_KEY": "qdrant-secret",
            "VLM_SERVICE_QDRANT_HTTPS": "yes",
        }
    )
    assert config.api_key == "gateway-secret"
    assert config.media_root == Path("~/service-media").expanduser().resolve()
    assert config.output_root == Path("/tmp/service-output").resolve()
    assert config.allow_remote_images is True
    assert config.max_concurrency == 4
    assert config.public_metrics is False
    assert config.rate_limit == RateLimit(30, 3600)
    assert config.max_body_bytes == 2048
    assert config.runtime_config_path == Path("/tmp/runtime.yaml").resolve()
    assert config.vlm_provider == "openai-compatible"
    assert config.vlm_base_url == "https://vllm.internal:8000"
    assert config.vlm_model == "Qwen/Qwen3-VL-8B-Instruct"
    assert config.vlm_api_key == "provider-secret"
    assert config.model_size == "8b"
    assert config.request_timeout == 12.5
    assert config.store == "sqlite-vec"
    assert config.store_path == "indexes/main"
    assert config.qdrant_host == "qdrant.internal"
    assert config.qdrant_port == 6334
    assert config.qdrant_collection == "documents"
    assert config.qdrant_api_key == "qdrant-secret"
    assert config.qdrant_https is True


def test_blank_values_are_treated_as_unset() -> None:
    config = ServiceConfig.from_env(
        {
            "VLM_SERVICE_API_KEY": "   ",
            "VLM_SERVICE_MEDIA_ROOT": "",
            "VLM_SERVICE_MODEL_SIZE": " ",
        }
    )
    assert config.api_key is None
    assert config.media_root is None
    assert config.model_size is None


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
def test_boolean_accepts_true_spellings(value: str) -> None:
    config = ServiceConfig.from_env({"VLM_SERVICE_ALLOW_REMOTE_IMAGES": value})
    assert config.allow_remote_images is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off"])
def test_boolean_accepts_false_spellings(value: str) -> None:
    assert ServiceConfig.from_env({"VLM_SERVICE_ALLOW_REMOTE_IMAGES": value}).allow_remote_images is False


def test_invalid_boolean_names_the_environment_variable() -> None:
    with pytest.raises(InputValidationError, match=ALLOW_REMOTE_IMAGES_ENV):
        ServiceConfig.from_env({"VLM_SERVICE_ALLOW_REMOTE_IMAGES": "sometimes"})


def test_invalid_integer_names_the_environment_variable() -> None:
    with pytest.raises(InputValidationError, match=MAX_CONCURRENCY_ENV):
        ServiceConfig.from_env({"VLM_SERVICE_MAX_CONCURRENCY": "many"})


def test_non_positive_integer_is_rejected() -> None:
    with pytest.raises(InputValidationError, match=MAX_CONCURRENCY_ENV):
        ServiceConfig.from_env({"VLM_SERVICE_MAX_CONCURRENCY": "0"})


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("120/minute", RateLimit(120, 60)),
        ("5/second", RateLimit(5, 1)),
        ("2/hours", RateLimit(2, 3600)),
        (" 10 / minutes ", RateLimit(10, 60)),
    ],
)
def test_rate_limit_parsing(raw: str, expected: RateLimit) -> None:
    assert RateLimit.parse(raw) == expected


@pytest.mark.parametrize("raw", ["120", "120/day", "0/minute", "many/minute", "/minute"])
def test_rate_limit_rejects_invalid_values(raw: str) -> None:
    with pytest.raises(InputValidationError, match=RATE_LIMIT_ENV):
        RateLimit.parse(raw)


def test_rate_limit_rejects_invalid_window() -> None:
    with pytest.raises(InputValidationError):
        RateLimit(requests=10, window_seconds=0)


def test_max_body_bytes_default_and_override() -> None:
    assert ServiceConfig.from_env({"VLM_SERVICE_MAX_BODY_BYTES": "4096"}).max_body_bytes == 4096
    with pytest.raises(InputValidationError, match=MAX_BODY_BYTES_ENV):
        ServiceConfig.from_env({"VLM_SERVICE_MAX_BODY_BYTES": "-1"})


def test_request_timeout_validation() -> None:
    with pytest.raises(InputValidationError, match=REQUEST_TIMEOUT_ENV):
        ServiceConfig.from_env({"VLM_SERVICE_REQUEST_TIMEOUT": "soon"})
    with pytest.raises(InputValidationError, match=REQUEST_TIMEOUT_ENV):
        ServiceConfig.from_env({"VLM_SERVICE_REQUEST_TIMEOUT": "0"})


def test_model_path_is_resolved_and_made_absolute() -> None:
    config = ServiceConfig.from_env({"VLM_SERVICE_MODEL_PATH": "~/models/qwen"})
    assert config.model_path == Path("~/models/qwen").expanduser().resolve()


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(InputValidationError, match=VLM_PROVIDER_ENV):
        ServiceConfig.from_env({"VLM_SERVICE_VLM_PROVIDER": "vllm"})


@pytest.mark.parametrize(
    "environment",
    [
        {},
        {"VLM_SERVICE_VLM_MODEL": "model"},
        {"VLM_SERVICE_VLM_BASE_URL": "http://localhost:8000"},
    ],
)
def test_openai_compatible_provider_requires_base_url_and_model(
    environment: dict[str, str],
) -> None:
    with pytest.raises(InputValidationError, match="openai-compatible"):
        ServiceConfig.from_env({**environment, "VLM_SERVICE_VLM_PROVIDER": "openai-compatible"})


def test_openai_compatible_requires_http_scheme() -> None:
    with pytest.raises(InputValidationError, match=VLM_BASE_URL_ENV):
        ServiceConfig.from_env(
            {
                "VLM_SERVICE_VLM_PROVIDER": "openai-compatible",
                "VLM_SERVICE_VLM_BASE_URL": "ftp://models",
                "VLM_SERVICE_VLM_MODEL": "model",
            }
        )


def test_remote_provider_settings_are_rejected_for_qwen() -> None:
    with pytest.raises(InputValidationError, match="openai-compatible"):
        ServiceConfig.from_env({"VLM_SERVICE_VLM_BASE_URL": "http://localhost:8000"})


def test_multiple_model_selectors_are_rejected() -> None:
    with pytest.raises(InputValidationError, match=MODEL_SIZE_ENV):
        ServiceConfig.from_env(
            {"VLM_SERVICE_MODEL_SIZE": "2b", "VLM_SERVICE_MODEL_ID": "Qwen/Qwen3-VL-2B-Instruct"}
        )
    with pytest.raises(InputValidationError, match=MODEL_PATH_ENV):
        ServiceConfig.from_env({"VLM_SERVICE_MODEL_SIZE": "2b", "VLM_SERVICE_MODEL_PATH": "/models/q"})


def test_invalid_store_configuration_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="VLM_SERVICE_STORE"):
        ServiceConfig.from_env({"VLM_SERVICE_STORE": "redis"})
    with pytest.raises(InputValidationError, match="STORE_PATH"):
        ServiceConfig.from_env({"VLM_SERVICE_STORE_PATH": "/absolute"})
    with pytest.raises(InputValidationError, match="STORE_PATH"):
        ServiceConfig.from_env({"VLM_SERVICE_STORE_PATH": "../escape"})
    with pytest.raises(InputValidationError, match="QDRANT_PORT"):
        ServiceConfig.from_env({"VLM_SERVICE_QDRANT_PORT": "70000"})
    with pytest.raises(InputValidationError, match="QDRANT_HTTPS"):
        ServiceConfig.from_env({"VLM_SERVICE_QDRANT_HTTPS": "maybe"})
    with pytest.raises(InputValidationError, match="QDRANT_HOST"):
        ServiceConfig(qdrant_host="")
    with pytest.raises(InputValidationError, match="QDRANT_COLLECTION"):
        ServiceConfig(qdrant_collection="")


def test_direct_construction_validates_field_types() -> None:
    with pytest.raises(InputValidationError):
        ServiceConfig(max_concurrency=0)
    with pytest.raises(InputValidationError):
        ServiceConfig(rate_limit="120/minute")  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        ServiceConfig(request_timeout=1)  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        ServiceConfig(media_root="/tmp/media")  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        ServiceConfig(model_size=2)  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        ServiceConfig(vlm_provider="openai-compatible", vlm_base_url="http://x")


def test_secrets_are_hidden_from_repr() -> None:
    config = ServiceConfig(
        api_key="gateway-secret",
        vlm_provider="openai-compatible",
        vlm_base_url="http://localhost:8000",
        vlm_model="m",
        vlm_api_key="provider-secret",
        qdrant_api_key="qdrant-secret",
    )
    rendered = repr(config)
    assert "gateway-secret" not in rendered
    assert "provider-secret" not in rendered
    assert "qdrant-secret" not in rendered


def test_empty_secret_is_rejected_when_constructed_directly() -> None:
    with pytest.raises(InputValidationError):
        ServiceConfig(api_key=" ")
    with pytest.raises(InputValidationError):
        ServiceConfig(
            vlm_provider="openai-compatible",
            vlm_base_url="http://localhost:8000",
            vlm_model="m",
            vlm_api_key=" ",
        )


def test_from_env_falls_back_to_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLM_SERVICE_MAX_CONCURRENCY", "7")
    assert ServiceConfig.from_env().max_concurrency == 7
