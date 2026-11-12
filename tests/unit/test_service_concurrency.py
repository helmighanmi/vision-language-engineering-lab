# Path: tests/unit/test_service_concurrency.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Unit tests for per-capability admission control."""

from __future__ import annotations

import pytest

from vlm_engineering.exceptions import InputValidationError
from vlm_engineering.service.concurrency import ConcurrencyLimiter, ServiceBusyError


def test_limiter_requires_a_positive_integer() -> None:
    for value in (0, -1, True, 1.5):
        with pytest.raises(InputValidationError):
            ConcurrencyLimiter(value)  # type: ignore[arg-type]


def test_slots_are_released_after_use() -> None:
    limiter = ConcurrencyLimiter(1)
    with limiter.slot("generation"):
        pass
    with limiter.slot("generation"):
        pass


def test_saturated_capability_raises_and_reports_itself() -> None:
    limiter = ConcurrencyLimiter(1)
    with limiter.slot("embedding"):
        with pytest.raises(ServiceBusyError) as caught:
            with limiter.slot("embedding"):
                pass
    assert caught.value.capability == "embedding"
    assert "embedding" in str(caught.value)


def test_capabilities_are_independent() -> None:
    limiter = ConcurrencyLimiter(1)
    with limiter.slot("generation"):
        with limiter.slot("embedding"):
            assert limiter.limit == 1


def test_release_happens_even_when_the_block_fails() -> None:
    limiter = ConcurrencyLimiter(1)
    with pytest.raises(RuntimeError, match="boom"):
        with limiter.slot("query"):
            raise RuntimeError("boom")
    with limiter.slot("query"):
        pass
