# Path: src/vlm_engineering/service/routers/query.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Retrieval-augmented question answering over the configured store."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Request

from ...exceptions import InputValidationError
from ..catalog import generation_model
from ..policy import resolve_media_reference
from ..schemas import QueryRequest, QueryResponse
from .dependencies import service_config, service_runtime

router = APIRouter(prefix="/v1", tags=["inference"])

NO_EVIDENCE_ANSWER = "No relevant evidence found."


def _item_text(item: Any) -> str:
    if isinstance(item, dict):
        text = item.get("text") or item.get("title") or item.get("image")
        return str(text).strip()
    return str(item).strip()


def _answer_prompt(question: str, results: list[Any]) -> str:
    lines = [
        "Answer the question using only the sources below.",
        "If the sources are insufficient, say so explicitly.",
        "",
        f"QUESTION: {question}",
        "",
    ]
    for index, result in enumerate(results, 1):
        text = _item_text(result.item)
        if text:
            lines.append(f"SOURCE {index}: {text}")
    return "\n".join(lines)


def _limits(runtime: Any, payload: QueryRequest) -> tuple[int, int, int]:
    profile = runtime.runtime_profile
    rag = profile.rag if profile is not None else None
    top_k = payload.top_k if payload.top_k is not None else (rag.top_k if rag else 3)
    candidate_k = (
        payload.candidate_k if payload.candidate_k is not None else (rag.candidate_k if rag else 12)
    )
    max_new_tokens = (
        payload.max_new_tokens if payload.max_new_tokens is not None else (rag.max_new_tokens if rag else 256)
    )
    if candidate_k < top_k:
        raise InputValidationError("candidate_k must be greater than or equal to top_k.")
    return top_k, candidate_k, max_new_tokens


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    """Embed the question, retrieve evidence, rerank it and answer."""
    config = service_config(request)
    runtime = service_runtime(request)
    top_k, candidate_k, max_new_tokens = _limits(runtime, payload)
    model = generation_model(config)

    query_image = (
        resolve_media_reference(payload.query_image, config)
        if payload.query_image is not None
        else None
    )
    query_input: dict[str, str] = {"text": payload.question}
    if query_image is not None:
        query_input["image"] = query_image
    empty = QueryResponse(answer=NO_EVIDENCE_ANSWER, chunks=[], model=model)

    # One query slot covers the whole pipeline so a request cannot occupy
    # several capability semaphores at once and deadlock under load.
    with runtime.limiter.slot("query"):
        vectors = np.asarray(runtime.embedder.encode([query_input]), dtype=np.float64)
        if vectors.ndim != 2 or len(vectors) != 1:
            raise InputValidationError("Embedding provider returned an unexpected shape.")
        candidates = runtime.configured_store().search(vectors[0], top_k=candidate_k)
        if not candidates:
            return empty
        results = runtime.reranker.rerank(
            query_input,
            [candidate.item for candidate in candidates],
            top_k=top_k,
        )
        prompt = _answer_prompt(payload.question, results)
        if query_image is not None:
            answer = runtime.generator.generate(
                [query_image], prompt, max_new_tokens=max_new_tokens
            )
        else:
            answer = runtime.generator.generate_text(prompt, max_new_tokens=max_new_tokens)

    return QueryResponse(
        answer=answer,
        chunks=[
            {"item": result.item, "score": float(result.score), "rank": int(result.rank)}
            for result in results
        ],
        model=model,
    )
