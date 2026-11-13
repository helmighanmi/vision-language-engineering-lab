# Path: tests/unit/test_service_metrics.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Unit tests for the in-tree Prometheus metrics registry."""

from __future__ import annotations

from vlm_engineering.service.metrics import UNMATCHED, MetricsRegistry


class _Clock:
    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now


def test_empty_registry_still_exposes_gauges() -> None:
    content_type, text = MetricsRegistry().render()
    assert content_type.startswith("text/plain; version=0.0.4")
    assert "# TYPE vlm_service_requests_in_flight gauge" in text
    assert "vlm_service_requests_in_flight 0" in text
    assert "vlm_service_uptime_seconds" in text


def test_counters_and_latency_are_aggregated_per_route() -> None:
    registry = MetricsRegistry()
    registry.observe(method="POST", path="/v1/query", status=200, duration=0.1)
    registry.observe(method="POST", path="/v1/query", status=200, duration=0.3)
    registry.observe(method="POST", path="/v1/query", status=503, duration=0.2)
    _, text = registry.render()

    assert 'vlm_service_requests_total{method="POST",path="/v1/query",status="200"} 2' in text
    assert 'vlm_service_requests_total{method="POST",path="/v1/query",status="503"} 1' in text
    assert (
        'vlm_service_request_duration_seconds_count{method="POST",path="/v1/query"} 3'
        in text
    )
    assert (
        'vlm_service_request_duration_seconds_sum{method="POST",path="/v1/query"} 0.600000'
        in text
    )
    assert (
        'vlm_service_request_duration_max_seconds{method="POST",path="/v1/query"} 0.300000'
        in text
    )


def test_observations_are_sorted_for_stable_output() -> None:
    registry = MetricsRegistry()
    registry.observe(method="GET", path="/v1/models", status=200, duration=0.01)
    registry.observe(method="GET", path="/health", status=200, duration=0.01)
    _, text = registry.render()
    total_lines = [line for line in text.splitlines() if "/health" in line and "total" in line]
    models_lines = [line for line in text.splitlines() if "/v1/models" in line and "total" in line]
    assert text.index(total_lines[0]) < text.index(models_lines[0])


def test_in_flight_tracks_begin_and_finish() -> None:
    registry = MetricsRegistry()
    registry.begin()
    registry.begin()
    assert "vlm_service_requests_in_flight 2" in registry.render()[1]
    registry.finish()
    assert "vlm_service_requests_in_flight 1" in registry.render()[1]
    registry.finish()
    registry.finish()
    assert "vlm_service_requests_in_flight 0" in registry.render()[1]


def test_uptime_uses_the_injected_clock() -> None:
    clock = _Clock()
    registry = MetricsRegistry(clock=clock)
    clock.now += 12.5
    assert "vlm_service_uptime_seconds 12.500000" in registry.render()[1]


def test_label_values_are_escaped() -> None:
    registry = MetricsRegistry()
    registry.observe(method="GET", path='/v1/"quoted"\\path', status=200, duration=0.0)
    _, text = registry.render()
    assert 'path="/v1/\\"quoted\\"\\\\path"' in text


def test_unmatched_paths_share_one_label() -> None:
    registry = MetricsRegistry()
    registry.observe(method="GET", path=UNMATCHED, status=404, duration=0.0)
    registry.observe(method="GET", path=UNMATCHED, status=404, duration=0.0)
    _, text = registry.render()
    assert 'vlm_service_requests_total{method="GET",path="<unmatched>",status="404"} 2' in text
