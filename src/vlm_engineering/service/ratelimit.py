# Path: src/vlm_engineering/service/ratelimit.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Single-process token-bucket rate limiting.

``VLM_SERVICE_RATE_LIMIT`` (default ``120/minute``) becomes a bucket with that
many tokens, refilled at ``requests / window`` per second. Buckets are keyed by
the configured API key when the caller presents it correctly, otherwise by peer
IP, so an attacker cannot create one bucket per guessed key.

Liveness and readiness probes are exempt: a traffic flood must not make a pod
fail its own health checks. The limiter is process-local; it is not a substitute
for a gateway when running multiple workers or replicas.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from starlette.types import ASGIApp, Receive, Scope, Send

from .config import RateLimit, ServiceConfig
from .errors import build_error_response
from .security import extract_api_key

EXEMPT_PATHS = frozenset({"/health", "/ready"})
_MAX_TRACKED_BUCKETS = 4096
_SAFE_API_KEY_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~+/="
)


@dataclass
class _Bucket:
    tokens: float
    updated_at: float


class TokenBucketLimiter:
    """Fixed-capacity buckets refilled continuously."""

    def __init__(
        self,
        limit: RateLimit,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_buckets: int = _MAX_TRACKED_BUCKETS,
    ) -> None:
        self.limit = limit
        self.rate_per_second = limit.requests / limit.window_seconds
        self._clock = clock
        self._max_buckets = max_buckets
        self._lock = threading.Lock()
        self._buckets: dict[str, _Bucket] = {}

    def check(self, key: str) -> float:
        """Consume one token; return 0 when allowed or the wait in seconds."""
        now = self._clock()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                self._buckets[key] = _Bucket(self.limit.requests - 1.0, now)
                self._prune(now)
                return 0.0
            elapsed = max(now - bucket.updated_at, 0.0)
            bucket.tokens = min(self.limit.requests, bucket.tokens + elapsed * self.rate_per_second)
            bucket.updated_at = now
            if bucket.tokens < 1.0:
                return (1.0 - bucket.tokens) / self.rate_per_second
            bucket.tokens -= 1.0
            return 0.0

    def _prune(self, now: float) -> None:
        """Bound memory by dropping stale buckets, oldest first (lock held)."""
        if len(self._buckets) <= self._max_buckets:
            return
        stale_after = self.limit.window_seconds * 2
        stale = [
            key
            for key, bucket in self._buckets.items()
            if bucket.tokens >= self.limit.requests or now - bucket.updated_at > stale_after
        ]
        for key in stale:
            del self._buckets[key]
        overflow = len(self._buckets) - self._max_buckets
        if overflow > 0:
            oldest = sorted(
                self._buckets,
                key=lambda key: self._buckets[key].updated_at,
            )
            for key in oldest[:overflow]:
                del self._buckets[key]


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def _looks_like_key(value: str) -> bool:
    return 0 < len(value) <= 512 and all(character in _SAFE_API_KEY_CHARACTERS for character in value)


def rate_limit_subject(scope: Scope, config: ServiceConfig) -> str:
    """Return the bucket key: the presented valid API key, else the peer IP."""
    headers = {name.decode("latin-1"): value.decode("latin-1") for name, value in scope.get("headers", [])}
    presented = extract_api_key(headers)
    if config.api_key is not None and presented is not None and _looks_like_key(presented):
        if secrets.compare_digest(presented, config.api_key):
            return f"key:{_fingerprint(presented)}"
    client = scope.get("client")
    host = client[0] if client else "unknown"
    return f"ip:{host}"


class RateLimitMiddleware:
    """Reject over-budget requests with ``429`` and ``Retry-After``."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        config: ServiceConfig,
        limiter: TokenBucketLimiter | None = None,
    ) -> None:
        self.app = app
        self.config = config
        self.limiter = limiter or TokenBucketLimiter(config.rate_limit)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") in EXEMPT_PATHS:
            await self.app(scope, receive, send)
            return
        retry_after = self.limiter.check(rate_limit_subject(scope, self.config))
        if retry_after <= 0:
            await self.app(scope, receive, send)
            return
        response = build_error_response(
            429,
            "rate_limited",
            f"Rate limit of {self.limiter.limit.requests} requests per "
            f"{self.limiter.limit.window_seconds} seconds exceeded.",
        )
        response.headers["Retry-After"] = str(max(1, math.ceil(retry_after)))
        await response(scope, receive, send)
