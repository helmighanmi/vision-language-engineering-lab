# Path: src/vlm_engineering/service/routers/jobs.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Job status polling and deletion for asynchronous routes."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response, status

from ..schemas import JobStatusResponse
from .dependencies import job_registry

router = APIRouter(prefix="/v1", tags=["jobs"])


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str, request: Request) -> JobStatusResponse:
    """Return the current state of a submitted job."""
    job = job_registry(request).get(job_id)
    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        result=job.result,
        error=job.error,
    )


@router.delete("/jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_job(job_id: str, request: Request) -> Response:
    """Delete a finished job so its result is no longer retained."""
    job_registry(request).delete(job_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
