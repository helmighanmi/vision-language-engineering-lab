# Path: src/vlm_engineering/service/metrics.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""In-tree request metrics with Prometheus text exposition.

The registry is deliberately tiny: per-route counters, latency sum/count/max and
an in-flight gauge. Cardinality is bounded by using the route template (for
example ``/v1/jobs/{job_id}``) rather than the raw request path, and unmatched
requests are folded into a single ``<unmatched>`` label.

Counters live in one process. They reset on restart and are not shared between
workers, which matches the documented single-process deployment model.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

UNMATCHED = "<unmatched>"
_EXPOSITION = "text/plain; version=0.0.4; charset=utf-8"

_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
}


def _escape_label(value: str) -> str:
    return "".join(_ESCAPES.get(character, character) for character in value)


def _labels(method: str, path: str, status: str | None = None) -> str:
    parts = [f'method="{_escape_label(method)}"', f'path="{_escape_label(path)}"']
    if status is not None:
        parts.append(f'status="{_escape_label(status)}"')
    return ",".join(parts)


@dataclass
class _Latency:
    count: int = 0
    total: float = 0.0
    maximum: float = 0.0


class MetricsRegistry:
    """Thread-safe counters for the HTTP surface."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._started_at = clock()
        self._lock = threading.Lock()
        self._requests: dict[tuple[str, str, str], int] = {}
        self._latency: dict[tuple[str, str], _Latency] = {}
        self._in_flight = 0

    # -- Recording ---------------------------------------------------------

    def begin(self) -> None:
        """Count a request that is currently being handled."""
        with self._lock:
            self._in_flight += 1

    def finish(self) -> None:
        """Count a request that has just left the middleware."""
        with self._lock:
            self._in_flight = max(self._in_flight - 1, 0)

    def observe(self, *, method: str, path: str, status: int, duration: float) -> None:
        """Record one completed request."""
        status_label = str(status)
        with self._lock:
            exact = (method, path, status_label)
            self._requests[exact] = self._requests.get(exact, 0) + 1
            route = (method, path)
            entry = self._latency.get(route)
            if entry is None:
                entry = _Latency()
                self._latency[route] = entry
            entry.count += 1
            entry.total += duration
            entry.maximum = max(entry.maximum, duration)

    # -- Exposition --------------------------------------------------------

    def render(self) -> tuple[str, str]:
        """Return ``(content_type, prometheus_text)`` for the ``/metrics`` route."""
        with self._lock:
            requests = dict(self._requests)
            latency = {
                route: _Latency(entry.count, entry.total, entry.maximum)
                for route, entry in self._latency.items()
            }
            in_flight = self._in_flight
        uptime = max(self._clock() - self._started_at, 0.0)

        lines = [
            "# HELP vlm_service_requests_total HTTP requests handled, by route and status.",
            "# TYPE vlm_service_requests_total counter",
        ]
        for (method, path, status_label), count in sorted(requests.items()):
            lines.append(
                f"vlm_service_requests_total{{{_labels(method, path, status_label)}}} {count}"
            )

        lines.extend(
            [
                "# HELP vlm_service_request_duration_seconds HTTP request latency in seconds.",
                "# TYPE vlm_service_request_duration_seconds summary",
            ]
        )
        for (method, path), entry in sorted(latency.items()):
            labels = _labels(method, path)
            lines.append(
                f"vlm_service_request_duration_seconds_sum{{{labels}}} {entry.total:.6f}"
            )
            lines.append(
                f"vlm_service_request_duration_seconds_count{{{labels}}} {entry.count}"
            )

        lines.extend(
            [
                "# HELP vlm_service_request_duration_max_seconds Slowest observed request.",
                "# TYPE vlm_service_request_duration_max_seconds gauge",
            ]
        )
        for (method, path), entry in sorted(latency.items()):
            lines.append(
                f"vlm_service_request_duration_max_seconds{{{_labels(method, path)}}} "
                f"{entry.maximum:.6f}"
            )

        lines.extend(
            [
                "# HELP vlm_service_requests_in_flight Requests currently being handled.",
                "# TYPE vlm_service_requests_in_flight gauge",
                f"vlm_service_requests_in_flight {in_flight}",
                "# HELP vlm_service_uptime_seconds Seconds since the registry was created.",
                "# TYPE vlm_service_uptime_seconds gauge",
                f"vlm_service_uptime_seconds {uptime:.6f}",
            ]
        )
        return _EXPOSITION, "\n".join(lines) + "\n"
