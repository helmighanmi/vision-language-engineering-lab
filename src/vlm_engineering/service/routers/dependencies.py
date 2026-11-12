# Path: src/vlm_engineering/service/routers/dependencies.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Shared state accessors for route handlers."""

from __future__ import annotations

from fastapi import Request

from ..config import ServiceConfig
from ..runtime import ServiceRuntime


def service_config(request: Request) -> ServiceConfig:
    config: ServiceConfig = request.app.state.config
    return config


def service_runtime(request: Request) -> ServiceRuntime:
    runtime: ServiceRuntime = request.app.state.runtime
    return runtime
