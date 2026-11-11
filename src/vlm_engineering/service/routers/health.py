# Path: src/vlm_engineering/service/routers/health.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Liveness and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def liveness() -> HealthResponse:
    """Report process liveness without touching model backends."""
    return HealthResponse()


@router.get("/ready", response_model=HealthResponse)
def readiness() -> HealthResponse:
    """Report service readiness; backend checks arrive in later phases."""
    return HealthResponse()
