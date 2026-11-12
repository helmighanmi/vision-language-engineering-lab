# Path: src/vlm_engineering/service/concurrency.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Per-capability admission control for blocking model calls.

Sync route handlers run on Starlette's threadpool. Each capability gets its own
bounded semaphore so a slow generation cannot starve embeddings. When a slot is
unavailable the request fails fast with ``ServiceBusyError`` (HTTP 503) instead
of queueing without bound.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

from ..exceptions import InputValidationError


class ServiceBusyError(RuntimeError):
    """Raised when a capability has no free concurrency slot."""

    def __init__(self, capability: str) -> None:
        super().__init__(f"{capability} capacity is exhausted; retry shortly.")
        self.capability = capability


class ConcurrencyLimiter:
    """One semaphore per capability, created lazily and shared per process."""

    def __init__(self, limit: int) -> None:
        if type(limit) is not int or limit <= 0:
            raise InputValidationError("max_concurrency must be a positive integer.")
        self.limit = limit
        self._lock = threading.Lock()
        self._slots: dict[str, threading.BoundedSemaphore] = {}

    def _semaphore(self, capability: str) -> threading.BoundedSemaphore:
        with self._lock:
            semaphore = self._slots.get(capability)
            if semaphore is None:
                semaphore = threading.BoundedSemaphore(self.limit)
                self._slots[capability] = semaphore
            return semaphore

    @contextmanager
    def slot(self, capability: str) -> Iterator[None]:
        """Acquire a non-blocking slot; raise :class:`ServiceBusyError` if full."""
        semaphore = self._semaphore(capability)
        if not semaphore.acquire(blocking=False):
            raise ServiceBusyError(capability)
        try:
            yield
        finally:
            semaphore.release()
