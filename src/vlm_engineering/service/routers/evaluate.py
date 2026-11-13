# Path: src/vlm_engineering/service/routers/evaluate.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Retrieval evaluation as an asynchronous job.

Mirrors ``vlm-lab evaluate`` for the same inputs: an in-memory index is built
from the supplied chunks with the service embedder, each case is retrieved at
``max(k_values)``, and the aggregate report keeps the CLI's metric names.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Request, status

from ...documents.schemas import VisualChunk
from ...evaluation import EvaluationCase, evaluate
from ...exceptions import InputValidationError
from ...inputs import normalize_input
from ...retrieval.in_memory import InMemoryVectorIndex
from ...retrieval.multimodal_rag import chunk_input
from ..policy import resolve_media_reference
from ..schemas import EvaluateRequest, EvaluateResponse, JobResponse
from .dependencies import job_registry, service_runtime

router = APIRouter(prefix="/v1", tags=["evaluation"])


def _prepare_cases(payload: EvaluateRequest, config: Any) -> list[EvaluationCase]:
    """Parse cases and canonicalise any local query image."""
    cases: list[EvaluationCase] = []
    for record in payload.cases:
        case = EvaluationCase.from_dict(record)
        if case.query_image is not None:
            case = replace(case, query_image=resolve_media_reference(case.query_image, config))
        cases.append(case)
    return cases


def _prepare_chunks(payload: EvaluateRequest, config: Any) -> tuple[list[VisualChunk], list[dict[str, str]]]:
    """Parse chunks and build their embedding inputs, enforcing media policy."""
    chunks: list[VisualChunk] = []
    inputs: list[dict[str, str]] = []
    for record in payload.chunks:
        chunk = VisualChunk.from_dict(record)
        if chunk.image_ref:
            chunk = replace(chunk, image_ref=resolve_media_reference(chunk.image_ref, config))
        inputs.append(chunk_input(chunk))
        chunks.append(chunk)
    return chunks, inputs


def _run_evaluate(
    payload: EvaluateRequest,
    runtime: Any,
    cases: list[EvaluationCase],
    chunks: list[VisualChunk],
    inputs: list[dict[str, str]],
) -> dict[str, Any]:
    """Embed the corpus, retrieve per case and aggregate the metrics."""
    embedder = runtime.embedder
    k_values = tuple(payload.k_values)
    max_k = max(k_values)
    with runtime.limiter.slot("evaluate"):
        embeddings = np.asarray(embedder.encode(inputs, prompt=payload.prompt), dtype=np.float64)
        if embeddings.ndim != 2 or len(embeddings) != len(inputs):
            raise InputValidationError("Embedding provider returned an unexpected shape.")
        index = InMemoryVectorIndex()
        index.add(embeddings, chunks)

        def retrieve(case: EvaluationCase) -> list[Any]:
            vector = embedder.encode(
                [normalize_input(case.retrieval_input())], prompt=payload.prompt
            )[0]
            return [result.item for result in index.search(vector, top_k=max_k)]

        report = evaluate(retrieve, cases, k_values=k_values)

    aggregate = dict(report.aggregate)
    latency_ms = float(aggregate.pop("latency_ms_mean", 0.0))
    response = EvaluateResponse(
        aggregate=aggregate,
        case_count=len(report.cases),
        k_values=list(report.k_values),
        latency_ms_mean=latency_ms,
    )
    return response.model_dump()


@router.post("/evaluate", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def evaluate_route(
    payload: EvaluateRequest,
    request: Request,
    background: BackgroundTasks,
) -> JobResponse:
    """Queue a retrieval evaluation run and return a job id."""
    config = request.app.state.config
    runtime = service_runtime(request)
    registry = job_registry(request)
    cases = _prepare_cases(payload, config)
    chunks, inputs = _prepare_chunks(payload, config)
    job = registry.create("evaluate")
    background.add_task(
        registry.run,
        job.job_id,
        partial(_run_evaluate, payload, runtime, cases, chunks, inputs),
    )
    return JobResponse(job_id=job.job_id)
