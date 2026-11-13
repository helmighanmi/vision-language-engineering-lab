# Path: tests/unit/test_service_jobs.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Unit tests for the in-process job registry."""

from __future__ import annotations

import pytest

from vlm_engineering.exceptions import InputValidationError
from vlm_engineering.service.concurrency import ServiceBusyError
from vlm_engineering.service.jobs import (
    ACCEPTED,
    COMPLETED,
    FAILED,
    RUNNING,
    JobNotFoundError,
    JobRegistry,
    JobStateError,
)


class _Clock:
    """Deterministic monotonic clock for TTL and eviction tests."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_registry_rejects_invalid_limits() -> None:
    with pytest.raises(InputValidationError):
        JobRegistry(ttl_seconds=0)
    with pytest.raises(InputValidationError):
        JobRegistry(max_jobs=0)
    with pytest.raises(InputValidationError):
        JobRegistry(max_jobs=True)  # type: ignore[arg-type]


def test_created_jobs_are_accepted_with_random_ids() -> None:
    registry = JobRegistry()
    first = registry.create("ingest")
    second = registry.create("index")
    assert first.status == ACCEPTED
    assert first.kind == "ingest"
    assert len(first.job_id) == 32
    assert first.job_id != second.job_id
    assert int(first.job_id, 16) >= 0
    assert registry.get(first.job_id).kind == "ingest"


def test_create_requires_a_kind() -> None:
    with pytest.raises(InputValidationError):
        JobRegistry().create("   ")


def test_run_records_the_result() -> None:
    registry = JobRegistry()
    job = registry.create("index")
    assert registry.run(job.job_id, lambda: {"indexed": 2}) is None
    finished = registry.get(job.job_id)
    assert finished.status == COMPLETED
    assert finished.result == {"indexed": 2}
    assert finished.error is None


def test_get_returns_a_snapshot() -> None:
    registry = JobRegistry()
    job = registry.create("index")
    registry.run(job.job_id, lambda: {"indexed": 1})
    snapshot = registry.get(job.job_id)
    assert snapshot.result is not None
    snapshot.result["indexed"] = 99
    assert registry.get(job.job_id).result == {"indexed": 1}


def test_run_marks_the_job_running_while_work_executes() -> None:
    registry = JobRegistry()
    job = registry.create("evaluate")
    observed: list[str] = []

    def work() -> dict[str, int]:
        observed.append(registry.get(job.job_id).status)
        return {"case_count": 0}

    registry.run(job.job_id, work)
    assert observed == [RUNNING]
    assert registry.get(job.job_id).status == COMPLETED


def test_package_errors_keep_their_message() -> None:
    registry = JobRegistry()
    job = registry.create("ingest")

    def work() -> dict[str, int]:
        raise InputValidationError("no images found")

    registry.run(job.job_id, work)
    failed = registry.get(job.job_id)
    assert failed.status == FAILED
    assert failed.error == "no images found"
    assert failed.result is None


def test_unexpected_errors_do_not_leak_internals() -> None:
    registry = JobRegistry()
    job = registry.create("ingest")

    def work() -> dict[str, int]:
        raise RuntimeError("secret-token in traceback")

    registry.run(job.job_id, work)
    failed = registry.get(job.job_id)
    assert failed.status == FAILED
    assert failed.error == "Job failed unexpectedly (RuntimeError)."
    assert "secret-token" not in str(failed.error)


def test_run_requires_an_existing_job() -> None:
    with pytest.raises(JobNotFoundError):
        JobRegistry().run("missing", lambda: {})


def test_finished_jobs_expire_after_the_ttl() -> None:
    clock = _Clock()
    registry = JobRegistry(ttl_seconds=30.0, clock=clock)
    job = registry.create("index")
    registry.run(job.job_id, lambda: {"indexed": 0})

    clock.advance(29.0)
    assert registry.get(job.job_id).status == COMPLETED

    clock.advance(2.0)
    with pytest.raises(JobNotFoundError):
        registry.get(job.job_id)
    assert len(registry) == 0


def test_running_jobs_never_expire() -> None:
    clock = _Clock()
    registry = JobRegistry(ttl_seconds=1.0, clock=clock)
    job = registry.create("evaluate")
    clock.advance(3600.0)
    assert registry.get(job.job_id).status == ACCEPTED


def test_capacity_evicts_the_oldest_finished_job() -> None:
    clock = _Clock()
    registry = JobRegistry(max_jobs=2, clock=clock)
    first = registry.create("index")
    registry.run(first.job_id, lambda: {"indexed": 0})
    clock.advance(1.0)
    second = registry.create("index")
    registry.run(second.job_id, lambda: {"indexed": 0})
    clock.advance(1.0)

    third = registry.create("index")
    assert len(registry) == 2
    with pytest.raises(JobNotFoundError):
        registry.get(first.job_id)
    assert registry.get(second.job_id).status == COMPLETED
    assert registry.get(third.job_id).status == ACCEPTED


def test_capacity_rejects_submissions_when_all_jobs_are_active() -> None:
    registry = JobRegistry(max_jobs=1)
    registry.create("ingest")
    with pytest.raises(ServiceBusyError, match="jobs"):
        registry.create("ingest")
    assert len(registry) == 1


def test_delete_removes_only_finished_jobs() -> None:
    registry = JobRegistry()
    job = registry.create("index")
    with pytest.raises(JobStateError) as caught:
        registry.delete(job.job_id)
    assert caught.value.status == ACCEPTED

    registry.run(job.job_id, lambda: {"indexed": 0})
    registry.delete(job.job_id)
    with pytest.raises(JobNotFoundError):
        registry.get(job.job_id)


def test_delete_and_get_unknown_jobs_raise() -> None:
    registry = JobRegistry()
    with pytest.raises(JobNotFoundError):
        registry.delete("missing")
    with pytest.raises(JobNotFoundError):
        registry.get("missing")
