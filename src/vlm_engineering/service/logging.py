# Path: src/vlm_engineering/service/logging.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""One JSON log line per request plus a correlated ``X-Request-ID``.

The middleware is deliberately outside rate limiting, authentication and the
route handlers so rejected requests are logged and measured too. It never logs
headers, bodies or API keys.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .metrics import UNMATCHED, MetricsRegistry

LOGGER_NAME = "vlm_engineering.service"
REQUEST_ID_HEADER = "X-Request-ID"
_MAX_REQUEST_ID_LENGTH = 64
_VALID_REQUEST_ID = re.compile(rf"^[A-Za-z0-9._-]{{1,{_MAX_REQUEST_ID_LENGTH}}}$")


def new_request_id(incoming: str | None) -> str:
    """Reuse a safe caller-supplied id, otherwise mint a random one."""
    if incoming is not None and _VALID_REQUEST_ID.match(incoming):
        return incoming
    return uuid.uuid4().hex


def configure_json_logging(level: int = logging.INFO) -> logging.Logger:
    """Install a message-only stream handler on the service logger once."""
    logger = logging.getLogger(LOGGER_NAME)
    if not any(
        isinstance(handler, logging.StreamHandler) and handler.get_name() == "vlm-service"
        for handler in logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.set_name("vlm-service")
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def _route_path(scope: Scope) -> str:
    route = scope.get("route")
    path = getattr(route, "path", None)
    return path if isinstance(path, str) and path else UNMATCHED


class RequestContextMiddleware:
    """Assign a request id, emit one JSON log line and record metrics."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        metrics: MetricsRegistry,
        logger: logging.Logger | None = None,
    ) -> None:
        self.app = app
        self.metrics = metrics
        self.logger = logger or logging.getLogger(LOGGER_NAME)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        incoming = Headers(scope=scope).get(REQUEST_ID_HEADER)
        request_id = new_request_id(incoming)
        state = scope.setdefault("state", {})
        state["request_id"] = request_id
        method = scope.get("method", "GET")
        started = time.perf_counter()
        status_code = 500
        self.metrics.begin()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = MutableHeaders(scope=message)
                headers[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            duration = time.perf_counter() - started
            path = _route_path(scope)
            self.metrics.finish()
            self.metrics.observe(
                method=method,
                path=path,
                status=status_code,
                duration=duration,
            )
            self._log(
                request_id=request_id,
                method=method,
                path=path,
                status=status_code,
                duration=duration,
            )

    def _log(
        self,
        *,
        request_id: str,
        method: str,
        path: str,
        status: int,
        duration: float,
    ) -> None:
        payload: dict[str, Any] = {
            "event": "request",
            "request_id": request_id,
            "method": method,
            "path": path,
            "status": status,
            "duration_ms": round(duration * 1000.0, 3),
        }
        self.logger.info(json.dumps(payload, separators=(",", ":")))
