# Path: tests/unit/test_service_logging.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Unit tests for request ids, structured logs and request metrics."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vlm_engineering.service.logging import (
    LOGGER_NAME,
    RequestContextMiddleware,
    configure_json_logging,
    new_request_id,
)
from vlm_engineering.service.metrics import MetricsRegistry


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def messages(self) -> list[dict[str, Any]]:
        return [json.loads(record.getMessage()) for record in self.records]


async def _call(
    app: ASGIApp,
    scope: Scope,
) -> tuple[int, dict[str, str]]:
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope, receive, send)
    start = next(message for message in sent if message["type"] == "http.response.start")
    headers = {
        name.decode("latin-1").lower(): value.decode("latin-1")
        for name, value in start.get("headers", [])
    }
    return int(start["status"]), headers


async def _app(scope: Scope, receive: Receive, send: Send) -> None:
    await send({"type": "http.response.start", "status": 204, "headers": []})
    await send({"type": "http.response.body", "body": b""})


def _scope(headers: list[tuple[bytes, bytes]] | None = None) -> Scope:
    return {
        "type": "http",
        "method": "POST",
        "path": "/v1/query",
        "headers": headers or [],
        "client": ("10.0.0.1", 1234),
    }


def test_new_request_id_accepts_safe_values() -> None:
    assert new_request_id("abc-123_XYZ.9") == "abc-123_XYZ.9"


def test_new_request_id_replaces_unsafe_values() -> None:
    for unsafe in ("", "has space", "semi;colon", "x" * 65, "é"):
        generated = new_request_id(unsafe)
        assert generated != unsafe
        assert re.fullmatch(r"[0-9a-f]{32}", generated)
    assert re.fullmatch(r"[0-9a-f]{32}", new_request_id(None))


def test_middleware_echoes_a_supplied_request_id_and_logs_json() -> None:
    capture = _Capture()
    logger = logging.getLogger("tests.request-context")
    logger.handlers = [capture]
    logger.setLevel(logging.INFO)
    registry = MetricsRegistry()
    middleware = RequestContextMiddleware(_app, metrics=registry, logger=logger)
    scope = _scope([(b"x-request-id", b"req-1")])

    status, headers = asyncio.run(_call(middleware, scope))

    assert status == 204
    assert headers["x-request-id"] == "req-1"
    assert scope["state"]["request_id"] == "req-1"
    payload = capture.messages()[0]
    assert payload["event"] == "request"
    assert payload["request_id"] == "req-1"
    assert payload["method"] == "POST"
    assert payload["path"] == "<unmatched>"
    assert payload["status"] == 204
    assert isinstance(payload["duration_ms"], float)
    assert "x-request-id" not in json.dumps(payload)  # no headers are logged


def test_middleware_mints_a_request_id_when_the_header_is_invalid() -> None:
    capture = _Capture()
    logger = logging.getLogger("tests.request-context-invalid")
    logger.handlers = [capture]
    logger.setLevel(logging.INFO)
    middleware = RequestContextMiddleware(_app, metrics=MetricsRegistry(), logger=logger)
    scope = _scope([(b"x-request-id", b"not valid")])

    _, headers = asyncio.run(_call(middleware, scope))

    assert re.fullmatch(r"[0-9a-f]{32}", headers["x-request-id"])
    assert capture.messages()[0]["request_id"] == headers["x-request-id"]


def test_middleware_uses_the_route_template_for_metrics() -> None:
    registry = MetricsRegistry()
    middleware = RequestContextMiddleware(_app, metrics=registry, logger=logging.getLogger("tests.silent"))
    scope = _scope()
    scope["route"] = type("Route", (), {"path": "/v1/jobs/{job_id}"})()

    asyncio.run(_call(middleware, scope))

    _, text = registry.render()
    assert (
        'vlm_service_requests_total{method="POST",path="/v1/jobs/{job_id}",status="204"} 1'
        in text
    )


def test_middleware_ignores_non_http_scopes() -> None:
    seen: list[str] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope["type"])

    middleware = RequestContextMiddleware(app, metrics=MetricsRegistry())
    asyncio.run(middleware({"type": "lifespan"}, None, None))  # type: ignore[arg-type]
    assert seen == ["lifespan"]


def test_configure_json_logging_is_idempotent() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    original = list(logger.handlers)
    logger.handlers = []
    try:
        first = configure_json_logging(logging.DEBUG)
        second = configure_json_logging()
        handlers = [handler for handler in second.handlers if handler.get_name() == "vlm-service"]
        assert first is second
        assert len(handlers) == 1
        assert handlers[0].formatter is not None
        assert handlers[0].formatter._fmt == "%(message)s"  # noqa: SLF001
        assert second.level == logging.INFO
    finally:
        logger.handlers = original
