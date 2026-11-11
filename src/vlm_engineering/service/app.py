# Path: src/vlm_engineering/service/app.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""FastAPI application factory for the service API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .routers import health
from .schemas import ErrorResponse, package_version

_SERVICE_TITLE = "Vision-Language Engineering Lab service"


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Hold service-wide resources; runtime wiring arrives in later phases."""
    yield


async def _validation_error_response(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Shape FastAPI validation errors into the standard ``ErrorResponse``."""
    parts = []
    for issue in exc.errors():
        location = ".".join(str(item) for item in issue.get("loc", ()))
        parts.append(f"{location}: {issue.get('msg', 'invalid value')}")
    detail = "; ".join(parts) if parts else "Invalid request body."
    payload = ErrorResponse(error="validation_error", detail=detail)
    return JSONResponse(status_code=422, content=payload.model_dump())


def create_app() -> FastAPI:
    """Build the FastAPI application with health routes and error shaping."""
    application = FastAPI(
        title=_SERVICE_TITLE,
        version=package_version(),
        lifespan=_lifespan,
        exception_handlers={RequestValidationError: _validation_error_response},
    )
    application.include_router(health.router)
    return application


app = create_app()
