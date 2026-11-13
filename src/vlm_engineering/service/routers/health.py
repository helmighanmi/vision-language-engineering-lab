# Path: src/vlm_engineering/service/routers/health.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Liveness, readiness and metrics endpoints.

None of these routes run inference. ``/ready`` only asks the vector store for its
current item count and reports which lazy providers have already been built, so
it stays cheap enough for a frequent probe.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse

from ..schemas import HealthResponse, ReadinessResponse, StoreReadiness
from ..security import require_api_key

router = APIRouter(tags=["health"])

_HTTP_503 = 503


@router.get("/health", response_model=HealthResponse)
def liveness() -> HealthResponse:
    """Report process liveness without touching model backends."""
    return HealthResponse()


@router.get("/ready", response_model=ReadinessResponse)
def readiness(request: Request) -> Response:
    """Report service readiness: built backends and store reachability."""
    config = request.app.state.config
    runtime = request.app.state.runtime
    store = StoreReadiness(kind=config.store, ready=True)
    try:
        runtime.configured_store().count()
    except Exception as exc:  # noqa: BLE001 - readiness must never raise
        store = StoreReadiness(kind=config.store, ready=False, detail=type(exc).__name__)
    payload = ReadinessResponse(
        status="ok" if store.ready else "degraded",
        backends=runtime.loaded_backends(),
        store=store,
    )
    return JSONResponse(
        status_code=200 if store.ready else _HTTP_503,
        content=payload.model_dump(),
    )


@router.get("/metrics", response_class=PlainTextResponse)
def metrics(request: Request) -> Response:
    """Expose the in-process counters in Prometheus text format."""
    config = request.app.state.config
    if not config.public_metrics:
        require_api_key(request)
    content_type, text = request.app.state.metrics.render()
    return PlainTextResponse(content=text, media_type=content_type)
