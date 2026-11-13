# Path: src/vlm_engineering/service/app.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""FastAPI application factory for the service API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import ServiceConfig
from .errors import register_error_handlers, validation_error_response
from .jobs import JobRegistry
from .routers import (
    describe,
    embed,
    evaluate,
    health,
    index,
    ingest,
    jobs,
    models,
    query,
    rerank,
)
from .runtime import ServiceRuntime
from .schemas import package_version

_SERVICE_TITLE = "Vision-Language Engineering Lab service"

# Kept as a private alias for existing tests and callers.
_validation_error_response = validation_error_response


@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Yield, then release providers and stores on shutdown."""
    try:
        yield
    finally:
        runtime = getattr(application.state, "runtime", None)
        if runtime is not None:
            runtime.close()


def create_app(config: ServiceConfig | None = None) -> FastAPI:
    """Build the FastAPI application with health, catalog routes and runtime wiring."""
    service_config = config or ServiceConfig.from_env()
    application = FastAPI(
        title=_SERVICE_TITLE,
        version=package_version(),
        lifespan=_lifespan,
    )
    register_error_handlers(application)
    application.state.config = service_config
    application.state.runtime = ServiceRuntime(service_config)
    application.state.jobs = JobRegistry(
        ttl_seconds=service_config.job_ttl,
        max_jobs=service_config.max_jobs,
    )
    application.include_router(health.router)
    application.include_router(models.router)
    application.include_router(describe.router)
    application.include_router(embed.router)
    application.include_router(rerank.router)
    application.include_router(query.router)
    application.include_router(ingest.router)
    application.include_router(index.router)
    application.include_router(evaluate.router)
    application.include_router(jobs.router)
    return application


app = create_app()
