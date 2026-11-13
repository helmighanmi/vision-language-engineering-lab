# Path: src/vlm_engineering/service/errors.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Map package exceptions onto the documented ``ErrorResponse`` contract.

Handlers never expose tracebacks or backend internals. Expected client errors
keep their concise message; unexpected failures return a random-less generic
500 body and are logged server-side.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..exceptions import (
    InputValidationError,
    ModelLoadError,
    OptionalDependencyError,
    VLMEngineeringError,
)
from .concurrency import ServiceBusyError
from .jobs import JobNotFoundError, JobStateError
from .schemas import ErrorResponse
from .security import AuthenticationError

logger = logging.getLogger(__name__)

ExceptionHandler = Callable[[Request, Any], Any]


def build_error_response(status_code: int, error: str, detail: str | None) -> JSONResponse:
    """Build the documented ``ErrorResponse`` payload for any status code."""
    payload = ErrorResponse(error=error, detail=detail)
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _error_response(status_code: int, error: str, detail: str | None) -> JSONResponse:
    return build_error_response(status_code, error, detail)


def _constant(
    status_code: int,
    code: str,
    *,
    expose_detail: bool = True,
) -> ExceptionHandler:
    async def handler(_request: Request, exc: Exception) -> JSONResponse:
        detail = str(exc) if expose_detail and str(exc) else None
        return _error_response(status_code, code, detail)

    return handler


async def validation_error_response(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Shape FastAPI validation errors into the standard ``ErrorResponse``."""
    parts = []
    for issue in exc.errors():
        location = ".".join(str(item) for item in issue.get("loc", ()))
        parts.append(f"{location}: {issue.get('msg', 'invalid value')}")
    detail = "; ".join(parts) if parts else "Invalid request body."
    return _error_response(422, "validation_error", detail)


async def busy_error_response(_request: Request, exc: Exception) -> JSONResponse:
    """Report exhausted capacity with a short retry hint."""
    response = _error_response(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "service_busy",
        str(exc),
    )
    response.headers["Retry-After"] = "1"
    return response


async def unauthorized_response(_request: Request, exc: Exception) -> JSONResponse:
    """Report a missing or invalid API key and advertise the auth scheme."""
    response = _error_response(
        status.HTTP_401_UNAUTHORIZED,
        "unauthorized",
        str(exc) or "Missing or invalid API key.",
    )
    response.headers["WWW-Authenticate"] = "Bearer"
    return response


async def unexpected_error_response(_request: Request, exc: Exception) -> JSONResponse:
    """Log the failure and return a generic 500 without leaking internals."""
    logger.error("Unhandled service error", exc_info=exc)
    return _error_response(
        status.HTTP_500_INTERNAL_SERVER_ERROR,
        "internal_error",
        None,
    )


def exception_handlers() -> dict[type[Exception], ExceptionHandler]:
    """Return the exception-to-response mapping used by ``create_app``."""
    return {
        RequestValidationError: validation_error_response,
        InputValidationError: _constant(status.HTTP_400_BAD_REQUEST, "invalid_request"),
        ValueError: _constant(status.HTTP_400_BAD_REQUEST, "invalid_request"),
        FileNotFoundError: _constant(status.HTTP_400_BAD_REQUEST, "file_not_found"),
        PermissionError: _constant(status.HTTP_400_BAD_REQUEST, "permission_denied"),
        OptionalDependencyError: _constant(
            status.HTTP_503_SERVICE_UNAVAILABLE, "dependency_unavailable"
        ),
        ImportError: _constant(status.HTTP_503_SERVICE_UNAVAILABLE, "dependency_unavailable"),
        ModelLoadError: _constant(status.HTTP_503_SERVICE_UNAVAILABLE, "model_unavailable"),
        ServiceBusyError: busy_error_response,
        AuthenticationError: unauthorized_response,
        JobNotFoundError: _constant(status.HTTP_404_NOT_FOUND, "not_found"),
        JobStateError: _constant(status.HTTP_409_CONFLICT, "job_conflict"),
        VLMEngineeringError: _constant(status.HTTP_400_BAD_REQUEST, "request_error"),
        Exception: unexpected_error_response,
    }


def register_error_handlers(application: FastAPI) -> None:
    """Install every handler on ``application``."""
    for exception_type, handler in exception_handlers().items():
        application.add_exception_handler(exception_type, handler)
