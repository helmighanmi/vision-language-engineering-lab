# Path: tests/unit/test_service_middleware.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Unit tests for the body cap and request deadline middleware."""

from __future__ import annotations

import asyncio
import json

import pytest
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vlm_engineering.service.middleware import (
    BodyLimitMiddleware,
    RequestTimeoutMiddleware,
)


async def _read_all(receive: Receive) -> bytes:
    chunks: list[bytes] = []
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            break
        chunks.append(message.get("body", b""))
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


async def _echo_app(scope: Scope, receive: Receive, send: Send) -> None:
    body = await _read_all(receive)
    scope["seen_body"] = body
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


async def _call(
    app: ASGIApp,
    scope: Scope,
    messages: list[Message],
) -> tuple[int, bytes]:
    sent: list[Message] = []
    pending = list(messages)

    async def receive() -> Message:
        if pending:
            return pending.pop(0)
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope, receive, send)
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return int(start["status"]), body


def _scope(
    *,
    content_length: bytes | None = None,
    path: str = "/v1/query",
) -> Scope:
    headers = [] if content_length is None else [(b"content-length", content_length)]
    return {
        "type": "http",
        "method": "POST",
        "path": path,
        "headers": headers,
        "client": ("10.0.0.1", 1234),
    }


def _body(*chunks: bytes) -> list[Message]:
    messages: list[Message] = []
    for index, chunk in enumerate(chunks):
        messages.append(
            {
                "type": "http.request",
                "body": chunk,
                "more_body": index < len(chunks) - 1,
            }
        )
    return messages


def test_body_within_the_declared_limit_reaches_the_app() -> None:
    middleware = BodyLimitMiddleware(_echo_app, max_bytes=10)
    status, body = asyncio.run(
        _call(middleware, _scope(content_length=b"4"), _body(b"data"))
    )
    assert status == 200
    assert body == b"ok"


def test_declared_content_length_over_the_limit_is_rejected() -> None:
    middleware = BodyLimitMiddleware(_echo_app, max_bytes=10)
    scope = _scope(content_length=b"11")
    status, body = asyncio.run(_call(middleware, scope, _body(b"x" * 11)))
    assert status == 413
    assert json.loads(body)["error"] == "payload_too_large"
    assert "seen_body" not in scope


def test_malformed_content_length_is_a_400() -> None:
    middleware = BodyLimitMiddleware(_echo_app, max_bytes=10)
    status, body = asyncio.run(
        _call(middleware, _scope(content_length=b"not-a-number"), _body(b"data"))
    )
    assert status == 400
    assert json.loads(body)["error"] == "invalid_request"


def test_chunked_body_is_buffered_and_replayed() -> None:
    middleware = BodyLimitMiddleware(_echo_app, max_bytes=10)
    scope = _scope()
    status, _ = asyncio.run(_call(middleware, scope, _body(b"abc", b"def")))
    assert status == 200
    assert scope["seen_body"] == b"abcdef"


def test_chunked_body_over_the_limit_is_rejected() -> None:
    middleware = BodyLimitMiddleware(_echo_app, max_bytes=4)
    scope = _scope()
    status, body = asyncio.run(_call(middleware, scope, _body(b"abc", b"def")))
    assert status == 413
    assert json.loads(body)["error"] == "payload_too_large"
    assert "seen_body" not in scope


def test_body_middleware_passes_through_non_http_scopes() -> None:
    seen: list[str] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope["type"])

    middleware = BodyLimitMiddleware(app, max_bytes=10)
    asyncio.run(middleware({"type": "lifespan"}, None, None))  # type: ignore[arg-type]
    assert seen == ["lifespan"]


def test_timeout_returns_504_when_no_response_started() -> None:
    async def slow(scope: Scope, receive: Receive, send: Send) -> None:
        await asyncio.sleep(1.0)

    middleware = RequestTimeoutMiddleware(slow, seconds=0.01)
    status, body = asyncio.run(_call(middleware, _scope(), _body(b"")))
    assert status == 504
    assert json.loads(body)["error"] == "request_timeout"


def test_timeout_allows_fast_responses() -> None:
    middleware = RequestTimeoutMiddleware(_echo_app, seconds=5.0)
    status, _ = asyncio.run(_call(middleware, _scope(), _body(b"data")))
    assert status == 200


def test_timeout_after_response_started_is_reraised() -> None:
    async def slow_after_start(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await asyncio.sleep(1.0)

    middleware = RequestTimeoutMiddleware(slow_after_start, seconds=0.01)
    with pytest.raises(TimeoutError):
        asyncio.run(_call(middleware, _scope(), _body(b"")))


def test_timeout_middleware_passes_through_non_http_scopes() -> None:
    seen: list[str] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope["type"])

    middleware = RequestTimeoutMiddleware(app, seconds=1.0)
    asyncio.run(middleware({"type": "lifespan"}, None, None))  # type: ignore[arg-type]
    assert seen == ["lifespan"]
