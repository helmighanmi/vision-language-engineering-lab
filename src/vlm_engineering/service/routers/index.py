# Path: src/vlm_engineering/service/routers/index.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Chunk indexing as an asynchronous job.

The request is validated and the target store is constructed before the job is
accepted, so an unreadable ``VLM_SERVICE_OUTPUT_ROOT``, a disabled remote
image, or a missing ``[stores]`` extra answers immediately instead of failing
silently inside the background task.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Request, status

from ...exceptions import InputValidationError
from ..policy import reject_model_overrides, resolve_media_reference, resolve_output_path
from ..schemas import IndexRequest, IndexResponse, JobResponse
from .dependencies import job_registry, service_config, service_runtime

router = APIRouter(prefix="/v1", tags=["indexing"])


def _reject_embed_overrides(payload: IndexRequest, configured_size: str | None) -> None:
    """Reject per-request embedding model selection.

    ``embed_model_size`` carries a default so only an explicitly supplied value
    is compared against ``VLM_SERVICE_MODEL_SIZE``; callers that omit it simply
    get the service-wide model.
    """
    reject_model_overrides(model_path=payload.embed_model_path)
    if "embed_model_size" not in payload.model_fields_set:
        return
    if payload.embed_model_size != (configured_size or "2b"):
        raise InputValidationError(
            "Per-request model selection is not supported; "
            "configure VLM_SERVICE_MODEL_SIZE, VLM_SERVICE_MODEL_ID or "
            "VLM_SERVICE_MODEL_PATH instead."
        )


def _chunk_inputs(chunks: list[dict[str, Any]], config: Any) -> list[dict[str, str]]:
    """Build one embedding input per chunk, validating any media reference."""
    inputs: list[dict[str, str]] = []
    for chunk in chunks:
        value: dict[str, str] = {}
        text = chunk.get("text")
        if isinstance(text, str) and text.strip():
            value["text"] = text
        image = chunk.get("image_ref") or chunk.get("image")
        if isinstance(image, str) and image.strip():
            value["image"] = resolve_media_reference(image, config)
        if not value:
            raise InputValidationError("each chunk needs non-empty text or an image reference.")
        inputs.append(value)
    return inputs


def _store_for(payload: IndexRequest, request: Request, output: Path) -> Any:
    """Construct (or reuse) the store named by the request."""
    runtime = service_runtime(request)
    config = service_config(request)
    if payload.store == "sqlite-vec":
        return runtime.store("sqlite-vec", path=output)
    if payload.store == "qdrant":
        return runtime.store(
            "qdrant",
            host=config.qdrant_host,
            port=config.qdrant_port,
            collection=config.qdrant_collection,
            api_key=config.qdrant_api_key,
            https=config.qdrant_https,
        )
    return runtime.store("memory")


def _run_index(
    payload: IndexRequest,
    store: Any,
    runtime: Any,
    inputs: list[dict[str, str]],
    ids: list[str],
    output: Path,
) -> dict[str, Any]:
    """Embed and persist chunks inside the background task."""
    with runtime.limiter.slot("index"):
        embeddings = np.asarray(runtime.embedder.encode(inputs), dtype=np.float64)
        if embeddings.ndim != 2 or len(embeddings) != len(inputs):
            raise InputValidationError("Embedding provider returned an unexpected shape.")
        store.add(embeddings, payload.chunks, ids=ids)
        store.persist(output)
    response = IndexResponse(indexed=len(inputs), store=payload.store, output=payload.output)
    return response.model_dump()


@router.post("/index", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
def index(
    payload: IndexRequest,
    request: Request,
    background: BackgroundTasks,
) -> JobResponse:
    """Queue embedding and indexing of chunks and return a job id."""
    config = service_config(request)
    runtime = service_runtime(request)
    registry = job_registry(request)
    _reject_embed_overrides(payload, config.model_size)
    output = resolve_output_path(payload.output, config)
    inputs = _chunk_inputs(payload.chunks, config)
    store = _store_for(payload, request, output)
    ids = [str(chunk.get("chunk_id") or f"chunk-{position}") for position, chunk in enumerate(payload.chunks)]
    job = registry.create("index")
    background.add_task(
        registry.run,
        job.job_id,
        partial(_run_index, payload, store, runtime, inputs, ids, output),
    )
    return JobResponse(job_id=job.job_id)
