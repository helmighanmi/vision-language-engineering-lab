# Path: src/vlm_engineering/service/middleware.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Transport guards: request body cap and per-request deadline.

Both run outside the router. The body cap answers ``413`` before FastAPI parses
anything, and the deadline answers ``504`` when a handler has not produced a
response yet. Cancelling a sync handler releases the request but not the worker
thread, so the timeout protects the caller, not the model call itself; that
caveat is documented with the setting.
"""

from __future__ import annotations

import asyncio

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .errors import build_error_response

_HTTP_413 = 413
_HTTP_504 = 504


class BodyLimitMiddleware:
    """Reject request bodies larger than ``max_bytes``."""

    def __init__(self, app: ASGIApp, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        known, declared = _content_length(scope)
        if known:
            if declared is None:
                await _reject(
                    scope,
                    send,
                    400,
                    "invalid_request",
                    "Content-Length must be an integer.",
                )
                return
            if declared > self.max_bytes:
                await self._too_large(scope, send)
                return
            await self.app(scope, receive, send)
            return

        # No Content-Length (chunked upload): buffer up to the cap, then replay.
        body, too_large = await _read_body(receive, self.max_bytes)
        if too_large:
            await self._too_large(scope, send)
            return
        replayed = False

        async def replay() -> Message:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        await self.app(scope, replay, send)

    async def _too_large(self, scope: Scope, send: Send) -> None:
        await _reject(
            scope,
            send,
            _HTTP_413,
            "payload_too_large",
            f"Request body exceeds {self.max_bytes} bytes.",
        )


class RequestTimeoutMiddleware:
    """Return ``504`` when no response has started before ``seconds``."""

    def __init__(self, app: ASGIApp, *, seconds: float) -> None:
        self.app = app
        self.seconds = seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = False

        async def track_start(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            async with asyncio.timeout(self.seconds):
                await self.app(scope, receive, track_start)
        except TimeoutError:
            if started:
                # The response is already on the wire; nothing safe to send.
                raise
            await _reject(
                scope,
                send,
                _HTTP_504,
                "request_timeout",
                f"Request exceeded the {self.seconds:g}s service deadline.",
            )


def _content_length(scope: Scope) -> tuple[bool, int | None]:
    """Return ``(present, value)``; ``value`` is ``None`` when malformed."""
    for name, value in scope.get("headers", []):
        if name.lower() == b"content-length":
            try:
                return True, max(int(value), 0)
            except ValueError:
                return True, None
    return False, None


async def _read_body(receive: Receive, max_bytes: int) -> tuple[bytes, bool]:
    chunks: list[bytes] = []
    total = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return b"", False
        chunk = message.get("body", b"")
        total += len(chunk)
        if total > max_bytes:
            return b"", True
        chunks.append(chunk)
        if not message.get("more_body", False):
            return b"".join(chunks), False


async def _reject(scope: Scope, send: Send, status_code: int, error: str, detail: str) -> None:
    response = build_error_response(status_code, error, detail)
    await response(scope, _no_receive, send)


async def _no_receive() -> Message:  # pragma: no cover - JSONResponse never reads
    return {"type": "http.disconnect"}
