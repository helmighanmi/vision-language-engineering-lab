# Path: src/vlm_engineering/service/jobs.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Bounded, TTL-based in-process job registry.

``ingest``, ``index`` and ``evaluate`` are long-running and return ``202`` with a
job id; the work itself runs as a FastAPI background task. This registry is
deliberately process-local:

- jobs are lost on restart and are invisible to other workers or replicas;
- the number of retained jobs is capped (``VLM_SERVICE_MAX_JOBS``);
- finished jobs expire after ``VLM_SERVICE_JOB_TTL`` seconds.

Job ids are random UUID4 hex strings so they cannot be guessed or enumerated.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from ..exceptions import InputValidationError, VLMEngineeringError
from .concurrency import ServiceBusyError
from .config import DEFAULT_JOB_TTL, DEFAULT_MAX_JOBS

ACCEPTED = "accepted"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
TERMINAL_STATES = frozenset({COMPLETED, FAILED})

Work = Callable[[], dict[str, Any]]


class JobNotFoundError(LookupError):
    """Raised when a job id is unknown, expired, or already deleted."""

    def __init__(self, job_id: str) -> None:
        super().__init__(f"Unknown job: {job_id}.")
        self.job_id = job_id


class JobStateError(RuntimeError):
    """Raised when an operation is invalid for the job's current state."""

    def __init__(self, job_id: str, status: str) -> None:
        super().__init__(
            f"Job {job_id} is {status}; wait for it to finish before deleting it."
        )
        self.job_id = job_id
        self.status = status


@dataclass
class Job:
    """One submitted unit of work and its outcome."""

    job_id: str
    kind: str
    status: str = ACCEPTED
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0


class JobRegistry:
    """Thread-safe registry shared by every route in one process.

    FastAPI runs sync background tasks on its threadpool while status polls are
    served from other threads, so every read and write takes the same lock.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_JOB_TTL,
        max_jobs: int = DEFAULT_MAX_JOBS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(ttl_seconds) is not float or ttl_seconds <= 0:
            raise InputValidationError("job ttl must be a positive number of seconds.")
        if type(max_jobs) is not int or max_jobs <= 0:
            raise InputValidationError("max_jobs must be a positive integer.")
        self.ttl_seconds = ttl_seconds
        self.max_jobs = max_jobs
        self._clock = clock
        self._lock = threading.RLock()
        self._jobs: dict[str, Job] = {}

    # -- Queries -----------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            self._prune()
            return len(self._jobs)

    def get(self, job_id: str) -> Job:
        """Return a job snapshot, raising when it is unknown or expired."""
        with self._lock:
            self._prune()
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            return Job(
                job_id=job.job_id,
                kind=job.kind,
                status=job.status,
                result=None if job.result is None else dict(job.result),
                error=job.error,
                created_at=job.created_at,
                updated_at=job.updated_at,
            )

    # -- Mutations ---------------------------------------------------------

    def create(self, kind: str) -> Job:
        """Register a new ``accepted`` job and return it."""
        if not isinstance(kind, str) or not kind.strip():
            raise InputValidationError("job kind must be a non-empty string.")
        with self._lock:
            now = self._clock()
            self._prune(now)
            self._make_room()
            job = Job(
                job_id=uuid.uuid4().hex,
                kind=kind.strip(),
                created_at=now,
                updated_at=now,
            )
            self._jobs[job.job_id] = job
            return replace(job)

    def delete(self, job_id: str) -> None:
        """Remove a finished job; active jobs must be awaited first."""
        with self._lock:
            self._prune()
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.status not in TERMINAL_STATES:
                raise JobStateError(job_id, job.status)
            del self._jobs[job_id]

    def run(self, job_id: str, work: Work) -> dict[str, Any] | None:
        """Execute ``work`` and record its result or failure.

        Background tasks must never raise: an unhandled exception would be
        logged by Starlette and leave the job stuck in ``running``. Expected
        package errors keep their message; anything else is reported as a
        generic failure so internals are not echoed back to pollers.
        """
        self._mark(job_id, RUNNING)
        try:
            result = work()
        except VLMEngineeringError as exc:
            self._finish(job_id, error=str(exc) or "Job failed.")
        except Exception as exc:  # noqa: BLE001 - the registry is the boundary
            self._finish(
                job_id,
                error=f"Job failed unexpectedly ({type(exc).__name__}).",
            )
        else:
            self._finish(job_id, result=result)
        return None

    # -- Internals ---------------------------------------------------------

    def _mark(self, job_id: str, status: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            job.status = status
            job.updated_at = self._clock()

    def _finish(
        self,
        job_id: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.status = COMPLETED if error is None else FAILED
            job.result = result
            job.error = error
            job.updated_at = self._clock()

    def _prune(self, now: float | None = None) -> None:
        """Drop finished jobs older than the TTL (callers hold the lock)."""
        current = self._clock() if now is None else now
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in TERMINAL_STATES
            and current - job.updated_at > self.ttl_seconds
        ]
        for job_id in expired:
            del self._jobs[job_id]

    def _make_room(self) -> None:
        """Evict the oldest finished job when the registry is at capacity."""
        while len(self._jobs) >= self.max_jobs:
            finished = [
                job for job in self._jobs.values() if job.status in TERMINAL_STATES
            ]
            if not finished:
                raise ServiceBusyError("jobs")
            oldest = min(finished, key=lambda job: job.updated_at)
            del self._jobs[oldest.job_id]
