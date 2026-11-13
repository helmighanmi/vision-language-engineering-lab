# Path: src/vlm_engineering/service/app.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""FastAPI application factory for the service API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from .config import ServiceConfig
from .errors import register_error_handlers, validation_error_response
from .jobs import JobRegistry
from .logging import RequestContextMiddleware, configure_json_logging
from .metrics import MetricsRegistry
from .middleware import BodyLimitMiddleware, RequestTimeoutMiddleware
from .ratelimit import RateLimitMiddleware
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
from .security import require_api_key

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
    """Build the FastAPI application with routes, guards and runtime wiring."""
    service_config = config or ServiceConfig.from_env()
    logger = configure_json_logging()
    metrics = MetricsRegistry()
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
    application.state.metrics = metrics

    # add_middleware order is inner-to-outer, so request ids, logging and
    # metrics wrap the guards and every route, including rejections.
    application.add_middleware(RequestTimeoutMiddleware, seconds=service_config.request_timeout)
    application.add_middleware(RateLimitMiddleware, config=service_config)
    application.add_middleware(BodyLimitMiddleware, max_bytes=service_config.max_body_bytes)
    application.add_middleware(RequestContextMiddleware, metrics=metrics, logger=logger)

    protected = [Depends(require_api_key)]
    application.include_router(health.router)
    for module in (models, describe, embed, rerank, query, ingest, index, evaluate, jobs):
        application.include_router(module.router, dependencies=protected)
    return application


app = create_app()
