# Path: src/vlm_engineering/service/config.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Environment-driven configuration for the HTTP service.

This module deliberately imports the standard library only: operators must be
able to validate configuration (and tests must be able to exercise it) without
installing the optional ``service`` extra. Pydantic/FastAPI never appear here;
the models provide their own validation at the request boundary.

Environment variables use the ``VLM_SERVICE_`` prefix. ``from_env`` accepts an
explicit mapping so tests do not need to mutate the process environment.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from ..exceptions import InputValidationError

API_KEY_ENV = "VLM_SERVICE_API_KEY"
MEDIA_ROOT_ENV = "VLM_SERVICE_MEDIA_ROOT"
OUTPUT_ROOT_ENV = "VLM_SERVICE_OUTPUT_ROOT"
ALLOW_REMOTE_IMAGES_ENV = "VLM_SERVICE_ALLOW_REMOTE_IMAGES"
MAX_CONCURRENCY_ENV = "VLM_SERVICE_MAX_CONCURRENCY"
PUBLIC_METRICS_ENV = "VLM_SERVICE_PUBLIC_METRICS"
RATE_LIMIT_ENV = "VLM_SERVICE_RATE_LIMIT"
MAX_BODY_BYTES_ENV = "VLM_SERVICE_MAX_BODY_BYTES"
RUNTIME_CONFIG_ENV = "VLM_SERVICE_RUNTIME_CONFIG"
VLM_PROVIDER_ENV = "VLM_SERVICE_VLM_PROVIDER"
VLM_BASE_URL_ENV = "VLM_SERVICE_VLM_BASE_URL"
VLM_MODEL_ENV = "VLM_SERVICE_VLM_MODEL"
VLM_API_KEY_ENV = "VLM_SERVICE_VLM_API_KEY"
MODEL_SIZE_ENV = "VLM_SERVICE_MODEL_SIZE"
MODEL_ID_ENV = "VLM_SERVICE_MODEL_ID"
MODEL_PATH_ENV = "VLM_SERVICE_MODEL_PATH"
REQUEST_TIMEOUT_ENV = "VLM_SERVICE_REQUEST_TIMEOUT"
STORE_ENV = "VLM_SERVICE_STORE"
STORE_PATH_ENV = "VLM_SERVICE_STORE_PATH"
QDRANT_HOST_ENV = "VLM_SERVICE_QDRANT_HOST"
QDRANT_PORT_ENV = "VLM_SERVICE_QDRANT_PORT"
QDRANT_COLLECTION_ENV = "VLM_SERVICE_QDRANT_COLLECTION"
QDRANT_API_KEY_ENV = "VLM_SERVICE_QDRANT_API_KEY"
QDRANT_HTTPS_ENV = "VLM_SERVICE_QDRANT_HTTPS"
JOB_TTL_ENV = "VLM_SERVICE_JOB_TTL"
MAX_JOBS_ENV = "VLM_SERVICE_MAX_JOBS"

DEFAULT_RATE_LIMIT = "120/minute"
DEFAULT_MAX_BODY_BYTES = 10 * 1024 * 1024
DEFAULT_REQUEST_TIMEOUT = 120.0
DEFAULT_MAX_CONCURRENCY = 2
DEFAULT_JOB_TTL = 900.0
DEFAULT_MAX_JOBS = 256
SUPPORTED_PROVIDERS = ("qwen", "openai-compatible")
SUPPORTED_STORES = ("memory", "sqlite-vec", "qdrant")

_RATE_LIMIT_PERIODS = {
    "second": 1,
    "seconds": 1,
    "minute": 60,
    "minutes": 60,
    "hour": 3600,
    "hours": 3600,
}
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _read(environ: Mapping[str, str], name: str) -> str | None:
    """Return a stripped environment value, treating blanks as unset."""
    value = environ.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_bool(raw: str, name: str) -> bool:
    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise InputValidationError(f"{name} must be a boolean (true/false, 1/0, yes/no, on/off).")


def _parse_int(raw: str, name: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise InputValidationError(f"{name} must be an integer.") from exc


def _parse_float(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise InputValidationError(f"{name} must be a number.") from exc
    if value <= 0:
        raise InputValidationError(f"{name} must be positive.")
    return value


def _parse_path(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


@dataclass(frozen=True)
class RateLimit:
    """A fixed-window request budget, e.g. ``120/minute``."""

    requests: int
    window_seconds: int

    def __post_init__(self) -> None:
        if type(self.requests) is not int or self.requests <= 0:
            raise InputValidationError("rate limit requests must be a positive integer.")
        if type(self.window_seconds) is not int or self.window_seconds <= 0:
            raise InputValidationError("rate limit window must be a positive integer.")

    @classmethod
    def parse(cls, value: str, *, name: str = RATE_LIMIT_ENV) -> "RateLimit":
        """Parse ``<requests>/<period>`` where period is seconds/minutes/hours."""
        requests_raw, separator, period_raw = value.partition("/")
        if not separator:
            raise InputValidationError(f"{name} must look like '120/minute'.")
        period = _RATE_LIMIT_PERIODS.get(period_raw.strip().lower())
        if period is None:
            raise InputValidationError(
                f"{name} period must be one of: second, minute, hour (or plural)."
            )
        requests = _parse_int(requests_raw.strip(), name)
        if requests <= 0:
            raise InputValidationError(f"{name} requests must be positive.")
        return cls(requests=requests, window_seconds=period)


@dataclass(frozen=True)
class ServiceConfig:
    """Validated service configuration resolved from ``VLM_SERVICE_*`` variables."""

    api_key: str | None = field(default=None, repr=False)
    media_root: Path | None = None
    output_root: Path | None = None
    allow_remote_images: bool = False
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    public_metrics: bool = True
    rate_limit: RateLimit = RateLimit(120, 60)
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES
    runtime_config_path: Path | None = None
    vlm_provider: str = "qwen"
    vlm_base_url: str | None = None
    vlm_model: str | None = None
    vlm_api_key: str | None = field(default=None, repr=False)
    model_size: str | None = None
    model_id: str | None = None
    model_path: Path | None = None
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    store: str = "memory"
    store_path: str = "index"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "vlm_engineering"
    qdrant_api_key: str | None = field(default=None, repr=False)
    qdrant_https: bool = False
    job_ttl: float = DEFAULT_JOB_TTL
    max_jobs: int = DEFAULT_MAX_JOBS

    def __post_init__(self) -> None:
        if self.api_key is not None:
            if not isinstance(self.api_key, str) or not self.api_key.strip():
                raise InputValidationError(f"{API_KEY_ENV} must not be empty.")
        for name, value in (("media_root", self.media_root), ("output_root", self.output_root)):
            if value is not None and not isinstance(value, Path):
                raise InputValidationError(f"{name} must be a Path when provided.")
        if self.runtime_config_path is not None and not isinstance(self.runtime_config_path, Path):
            raise InputValidationError("runtime_config_path must be a Path when provided.")
        if type(self.allow_remote_images) is not bool:
            raise InputValidationError(f"{ALLOW_REMOTE_IMAGES_ENV} must be a boolean.")
        if type(self.public_metrics) is not bool:
            raise InputValidationError(f"{PUBLIC_METRICS_ENV} must be a boolean.")
        if type(self.max_concurrency) is not int or self.max_concurrency <= 0:
            raise InputValidationError(f"{MAX_CONCURRENCY_ENV} must be a positive integer.")
        if type(self.max_body_bytes) is not int or self.max_body_bytes <= 0:
            raise InputValidationError(f"{MAX_BODY_BYTES_ENV} must be a positive integer.")
        if not isinstance(self.rate_limit, RateLimit):
            raise InputValidationError(f"{RATE_LIMIT_ENV} must be a RateLimit.")
        if type(self.request_timeout) is not float or self.request_timeout <= 0:
            raise InputValidationError(f"{REQUEST_TIMEOUT_ENV} must be a positive number.")
        if type(self.job_ttl) is not float or self.job_ttl <= 0:
            raise InputValidationError(f"{JOB_TTL_ENV} must be a positive number.")
        if type(self.max_jobs) is not int or self.max_jobs <= 0:
            raise InputValidationError(f"{MAX_JOBS_ENV} must be a positive integer.")
        if self.vlm_provider not in SUPPORTED_PROVIDERS:
            options = ", ".join(SUPPORTED_PROVIDERS)
            raise InputValidationError(f"{VLM_PROVIDER_ENV} must be one of: {options}.")
        self._validate_model_selection()
        self._validate_provider_settings()
        self._validate_store_settings()

    def _validate_store_settings(self) -> None:
        if self.store not in SUPPORTED_STORES:
            options = ", ".join(SUPPORTED_STORES)
            raise InputValidationError(f"{STORE_ENV} must be one of: {options}.")
        if not isinstance(self.store_path, str) or not self.store_path.strip():
            raise InputValidationError(f"{STORE_PATH_ENV} must be a non-empty string.")
        store_path = Path(self.store_path)
        if store_path.is_absolute() or ".." in store_path.parts:
            raise InputValidationError(f"{STORE_PATH_ENV} must be a relative path without '..'.")
        if not isinstance(self.qdrant_host, str) or not self.qdrant_host.strip():
            raise InputValidationError(f"{QDRANT_HOST_ENV} must be a non-empty string.")
        if type(self.qdrant_port) is not int or not 1 <= self.qdrant_port <= 65535:
            raise InputValidationError(f"{QDRANT_PORT_ENV} must be a port between 1 and 65535.")
        if not isinstance(self.qdrant_collection, str) or not self.qdrant_collection.strip():
            raise InputValidationError(f"{QDRANT_COLLECTION_ENV} must be a non-empty string.")
        if self.qdrant_api_key is not None and (
            not isinstance(self.qdrant_api_key, str) or not self.qdrant_api_key.strip()
        ):
            raise InputValidationError(f"{QDRANT_API_KEY_ENV} must not be empty.")
        if type(self.qdrant_https) is not bool:
            raise InputValidationError(f"{QDRANT_HTTPS_ENV} must be a boolean.")

    def _validate_model_selection(self) -> None:
        selected = [name for name, value in (
            (MODEL_SIZE_ENV, self.model_size),
            (MODEL_ID_ENV, self.model_id),
            (MODEL_PATH_ENV, self.model_path),
        ) if value is not None]
        if len(selected) > 1:
            raise InputValidationError(
                "Configure at most one of "
                f"{MODEL_SIZE_ENV}, {MODEL_ID_ENV}, {MODEL_PATH_ENV}."
            )
        if self.model_size is not None and (
            not isinstance(self.model_size, str) or not self.model_size.strip()
        ):
            raise InputValidationError(f"{MODEL_SIZE_ENV} must be a non-empty string.")
        if self.model_id is not None and (
            not isinstance(self.model_id, str) or not self.model_id.strip()
        ):
            raise InputValidationError(f"{MODEL_ID_ENV} must be a non-empty string.")
        if self.model_path is not None and not isinstance(self.model_path, Path):
            raise InputValidationError(f"{MODEL_PATH_ENV} must be a Path when provided.")

    def _validate_provider_settings(self) -> None:
        remote = (self.vlm_base_url, self.vlm_model, self.vlm_api_key)
        if self.vlm_provider == "qwen":
            if any(value is not None for value in remote):
                raise InputValidationError(
                    f"{VLM_BASE_URL_ENV}, {VLM_MODEL_ENV} and {VLM_API_KEY_ENV} "
                    f"are only valid with {VLM_PROVIDER_ENV}=openai-compatible."
                )
            return
        if not isinstance(self.vlm_base_url, str) or not self.vlm_base_url.strip():
            raise InputValidationError(
                f"{VLM_BASE_URL_ENV} is required when {VLM_PROVIDER_ENV}=openai-compatible."
            )
        scheme = urlparse(self.vlm_base_url).scheme
        if scheme not in ("http", "https"):
            raise InputValidationError(f"{VLM_BASE_URL_ENV} must be an http(s) URL.")
        if not isinstance(self.vlm_model, str) or not self.vlm_model.strip():
            raise InputValidationError(
                f"{VLM_MODEL_ENV} is required when {VLM_PROVIDER_ENV}=openai-compatible."
            )
        if self.vlm_api_key is not None and (
            not isinstance(self.vlm_api_key, str) or not self.vlm_api_key.strip()
        ):
            raise InputValidationError(f"{VLM_API_KEY_ENV} must not be empty.")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ServiceConfig":
        """Build a configuration from ``environ`` (defaults to ``os.environ``)."""
        source = os.environ if environ is None else environ
        rate_limit_raw = _read(source, RATE_LIMIT_ENV) or DEFAULT_RATE_LIMIT
        max_body_raw = _read(source, MAX_BODY_BYTES_ENV)
        timeout_raw = _read(source, REQUEST_TIMEOUT_ENV)
        return cls(
            api_key=_read(source, API_KEY_ENV),
            media_root=_parse_path(raw) if (raw := _read(source, MEDIA_ROOT_ENV)) else None,
            output_root=_parse_path(raw) if (raw := _read(source, OUTPUT_ROOT_ENV)) else None,
            allow_remote_images=_parse_bool(
                _read(source, ALLOW_REMOTE_IMAGES_ENV) or "false", ALLOW_REMOTE_IMAGES_ENV
            ),
            max_concurrency=_parse_int(
                _read(source, MAX_CONCURRENCY_ENV) or str(DEFAULT_MAX_CONCURRENCY),
                MAX_CONCURRENCY_ENV,
            ),
            public_metrics=_parse_bool(
                _read(source, PUBLIC_METRICS_ENV) or "true", PUBLIC_METRICS_ENV
            ),
            rate_limit=RateLimit.parse(rate_limit_raw),
            max_body_bytes=_parse_int(
                max_body_raw or str(DEFAULT_MAX_BODY_BYTES), MAX_BODY_BYTES_ENV
            ),
            runtime_config_path=(
                _parse_path(raw) if (raw := _read(source, RUNTIME_CONFIG_ENV)) else None
            ),
            vlm_provider=(_read(source, VLM_PROVIDER_ENV) or "qwen").lower(),
            vlm_base_url=_read(source, VLM_BASE_URL_ENV),
            vlm_model=_read(source, VLM_MODEL_ENV),
            vlm_api_key=_read(source, VLM_API_KEY_ENV),
            model_size=_read(source, MODEL_SIZE_ENV),
            model_id=_read(source, MODEL_ID_ENV),
            model_path=_parse_path(raw) if (raw := _read(source, MODEL_PATH_ENV)) else None,
            request_timeout=_parse_float(
                timeout_raw or str(DEFAULT_REQUEST_TIMEOUT), REQUEST_TIMEOUT_ENV
            ),
            store=(_read(source, STORE_ENV) or "memory").lower(),
            store_path=_read(source, STORE_PATH_ENV) or "index",
            qdrant_host=_read(source, QDRANT_HOST_ENV) or "localhost",
            qdrant_port=_parse_int(_read(source, QDRANT_PORT_ENV) or "6333", QDRANT_PORT_ENV),
            qdrant_collection=_read(source, QDRANT_COLLECTION_ENV) or "vlm_engineering",
            qdrant_api_key=_read(source, QDRANT_API_KEY_ENV),
            qdrant_https=_parse_bool(
                _read(source, QDRANT_HTTPS_ENV) or "false", QDRANT_HTTPS_ENV
            ),
            job_ttl=_parse_float(
                _read(source, JOB_TTL_ENV) or str(DEFAULT_JOB_TTL), JOB_TTL_ENV
            ),
            max_jobs=_parse_int(
                _read(source, MAX_JOBS_ENV) or str(DEFAULT_MAX_JOBS), MAX_JOBS_ENV
            ),
        )
