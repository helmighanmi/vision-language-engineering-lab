# Path: src/vlm_engineering/service/routers/ingest.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Image ingestion as an asynchronous job.

Ingestion is long-running (one VLM analysis per page), so the route validates
the request, queues a job and returns ``202`` immediately. Directory expansion
mirrors the CLI: one level deep, supported image suffixes only, and every
expanded file stays inside ``VLM_SERVICE_MEDIA_ROOT``.
"""

from __future__ import annotations

from functools import partial
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Request, status

from ...exceptions import OptionalDependencyError
from ..policy import resolve_media_entries
from ..schemas import IngestRequest, IngestResponse, IngestResultItem, JobResponse
from .dependencies import job_registry, service_config, service_runtime

router = APIRouter(prefix="/v1", tags=["ingestion"])

# Shape-compatible empty analysis so ``no_vlm`` ingestion still builds chunks.
_EMPTY_ANALYSIS = (
    '{"page_type":"mixed","title":"","summary":"","entities":[],'
    '"relations":[],"important_text":[],"uncertainties":[]}'
)


class _NullGenerator:
    """Deterministic stand-in used when ``no_vlm`` is requested."""

    def generate(self, _image: str, _prompt: str, *, max_new_tokens: int = 512) -> str:
        return _EMPTY_ANALYSIS


def _ingestion_pipeline() -> Any:
    """Import the pipeline lazily so a missing install maps to 503, not 500."""
    try:
        from ...ingestion.pipeline import IngestionPipeline
    except ImportError as exc:  # pragma: no cover - ingested extras are installed in tests
        raise OptionalDependencyError(
            'Ingestion needs the [ingest] extra: pip install "vision-language-engineering-lab[ingest]"'
        ) from exc
    return IngestionPipeline


def _run_ingest(
    pipeline_class: Any,
    payload: IngestRequest,
    runtime: Any,
    entries: list[str],
) -> dict[str, Any]:
    """Execute ingestion inside the background task and shape the response."""
    if payload.no_vlm:
        generator: Any = _NullGenerator()
    else:
        generator = runtime.generator
    pipeline = pipeline_class(
        generator,
        document_id=payload.document_id,
        source_file=payload.source_file,
    )
    with runtime.limiter.slot("ingest"):
        results = pipeline.ingest_images(entries)
    items = [
        IngestResultItem(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            source_file=chunk.source_file,
            page=chunk.page,
            chunk_type=chunk.chunk_type,
            title=chunk.title,
            text=chunk.text,
            image_ref=chunk.image_ref,
            errors=list(result.errors),
        )
        for result in results
        for chunk in result.chunks
    ]
    response = IngestResponse(
        results=items,
        total_chunks=len(items),
        total_errors=sum(len(result.errors) for result in results),
    )
    return response.model_dump()


@router.post("/ingest", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def ingest(
    payload: IngestRequest,
    request: Request,
    background: BackgroundTasks,
) -> JobResponse:
    """Queue ingestion of local images (or a directory) and return a job id."""
    config = service_config(request)
    runtime = service_runtime(request)
    registry = job_registry(request)
    pipeline_class = _ingestion_pipeline()
    entries = resolve_media_entries(payload.images, config)
    job = registry.create("ingest")
    background.add_task(
        registry.run,
        job.job_id,
        partial(_run_ingest, pipeline_class, payload, runtime, entries),
    )
    return JobResponse(job_id=job.job_id)
