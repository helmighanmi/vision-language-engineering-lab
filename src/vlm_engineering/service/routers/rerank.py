# Path: src/vlm_engineering/service/routers/rerank.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Candidate reranking."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, Request

from ..catalog import reranking_model
from ..policy import reject_model_overrides
from ..schemas import RerankRequest, RerankResponse, RerankResult
from .dependencies import service_config, service_runtime

router = APIRouter(prefix="/v1", tags=["inference"])


def _sigmoid(score: float) -> float:
    if score >= 0:
        return 1.0 / (1.0 + float(np.exp(-score)))
    exponent = float(np.exp(score))
    return exponent / (1.0 + exponent)


def _score(result: Any, normalize_scores: bool) -> float:
    value = float(result.score)
    # Providers configured with normalized scores already return (0, 1); only
    # apply the sigmoid when a raw score falls outside that range, so the flag
    # never double-normalizes.
    if normalize_scores and not 0.0 <= value <= 1.0:
        value = _sigmoid(value)
    return value


@router.post("/rerank", response_model=RerankResponse)
def rerank(payload: RerankRequest, request: Request) -> RerankResponse:
    """Score and order candidate documents for a query."""
    config = service_config(request)
    runtime = service_runtime(request)
    reject_model_overrides(
        model_size=payload.model_size,
        model_id=payload.model_id,
        model_path=payload.model_path,
    )
    with runtime.limiter.slot("reranking"):
        results = runtime.reranker.rerank(
            payload.query,
            payload.documents,
            top_k=payload.top_k,
            prompt=payload.prompt,
        )
    return RerankResponse(
        results=[
            RerankResult(
                item=result.item,
                score=_score(result, payload.normalize_scores),
                rank=int(result.rank),
            )
            for result in results
        ],
        model=reranking_model(config) or "unknown",
    )
