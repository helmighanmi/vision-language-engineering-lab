# Path: tests/unit/test_service_ratelimit.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Unit tests for the single-process token-bucket limiter."""

from __future__ import annotations

import asyncio
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from vlm_engineering.service.config import RateLimit, ServiceConfig
from vlm_engineering.service.ratelimit import (
    EXEMPT_PATHS,
    RateLimitMiddleware,
    TokenBucketLimiter,
    rate_limit_subject,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _scope(
    *,
    path: str = "/v1/models",
    headers: list[tuple[bytes, bytes]] | None = None,
    client: tuple[str, int] | None = ("10.0.0.1", 4321),
) -> Scope:
    scope: dict[str, Any] = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": headers or [],
    }
    if client is not None:
        scope["client"] = client
    return scope


async def _call(
    app: ASGIApp,
    scope: Scope,
) -> tuple[int, dict[str, str], bytes]:
    sent: list[Message] = []

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        sent.append(message)

    await app(scope, receive, send)
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    headers = {
        name.decode("latin-1").lower(): value.decode("latin-1")
        for name, value in start.get("headers", [])
    }
    return int(start["status"]), headers, body


async def _ok_app(scope: Scope, receive: Receive, send: Send) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def test_bucket_allows_the_budget_then_blocks() -> None:
    clock = _Clock()
    limiter = TokenBucketLimiter(RateLimit(2, 60), clock=clock)
    assert limiter.check("subject") == 0.0
    assert limiter.check("subject") == 0.0
    assert limiter.check("subject") > 0.0


def test_bucket_refills_continuously() -> None:
    clock = _Clock()
    limiter = TokenBucketLimiter(RateLimit(2, 60), clock=clock)
    limiter.check("subject")
    limiter.check("subject")
    blocked = limiter.check("subject")
    assert blocked == 30.0
    clock.now += 30.0
    assert limiter.check("subject") == 0.0
    assert limiter.check("subject") > 0.0


def test_buckets_are_isolated_per_subject() -> None:
    limiter = TokenBucketLimiter(RateLimit(1, 60))
    assert limiter.check("a") == 0.0
    assert limiter.check("a") > 0.0
    assert limiter.check("b") == 0.0


def test_full_or_idle_buckets_are_pruned_when_the_cap_is_reached() -> None:
    clock = _Clock()
    limiter = TokenBucketLimiter(RateLimit(1, 10), clock=clock, max_buckets=1)
    limiter.check("a")
    clock.now += 25.0  # stale: older than two windows
    limiter.check("b")
    assert "a" not in limiter._buckets  # noqa: SLF001 - prune contract
    assert "b" in limiter._buckets  # noqa: SLF001


def test_prune_keeps_recent_buckets_under_the_cap() -> None:
    clock = _Clock()
    limiter = TokenBucketLimiter(RateLimit(5, 60), clock=clock, max_buckets=8)
    for index in range(5):
        limiter.check(f"subject-{index}")
    limiter.check("fresh")
    assert len(limiter._buckets) == 6  # noqa: SLF001 - under the cap, nothing pruned


def test_prune_evicts_the_oldest_bucket_when_everything_is_active() -> None:
    clock = _Clock()
    limiter = TokenBucketLimiter(RateLimit(5, 60), clock=clock, max_buckets=2)
    limiter.check("oldest")
    clock.now += 1.0
    limiter.check("newer")
    clock.now += 1.0
    limiter.check("newest")
    assert set(limiter._buckets) == {"newer", "newest"}  # noqa: SLF001


def test_rate_limit_subject_prefers_a_valid_api_key() -> None:
    config = ServiceConfig(api_key="secret")
    scope = _scope(headers=[(b"x-api-key", b"secret")])
    subject = rate_limit_subject(scope, config)
    assert subject.startswith("key:")
    assert "secret" not in subject
    same = rate_limit_subject(scope, config)
    assert subject == same


def test_rate_limit_subject_falls_back_to_the_peer_ip() -> None:
    config = ServiceConfig(api_key="secret")
    wrong = _scope(headers=[(b"x-api-key", b"nope")])
    assert rate_limit_subject(wrong, config) == "ip:10.0.0.1"
    anonymous = ServiceConfig()
    valid_but_disabled = _scope(headers=[(b"x-api-key", b"secret")])
    assert rate_limit_subject(valid_but_disabled, anonymous) == "ip:10.0.0.1"
    assert rate_limit_subject(_scope(client=None), anonymous) == "ip:unknown"


def test_rate_limit_subject_ignores_non_ascii_keys() -> None:
    config = ServiceConfig(api_key="clé")
    scope = _scope(headers=[(b"x-api-key", "clé".encode("latin-1"))])
    assert rate_limit_subject(scope, config) == "ip:10.0.0.1"


def test_middleware_exempts_health_and_ready() -> None:
    limiter = TokenBucketLimiter(RateLimit(1, 60))
    middleware = RateLimitMiddleware(_ok_app, config=ServiceConfig(), limiter=limiter)
    for path in EXEMPT_PATHS:
        for _ in range(5):
            status, _, _ = asyncio.run(_call(middleware, _scope(path=path)))
            assert status == 200


def test_middleware_returns_429_with_retry_after() -> None:
    limiter = TokenBucketLimiter(RateLimit(1, 60))
    middleware = RateLimitMiddleware(_ok_app, config=ServiceConfig(), limiter=limiter)
    status, _, _ = asyncio.run(_call(middleware, _scope()))
    assert status == 200
    status, headers, body = asyncio.run(_call(middleware, _scope()))
    assert status == 429
    assert headers["retry-after"] == "60"
    assert b"rate_limited" in body
    assert b"Rate limit of 1 requests per 60 seconds exceeded." in body


def test_middleware_passes_through_non_http_scopes() -> None:
    seen: list[str] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope["type"])

    middleware = RateLimitMiddleware(app, config=ServiceConfig())
    asyncio.run(middleware({"type": "lifespan"}, None, None))  # type: ignore[arg-type]
    assert seen == ["lifespan"]
