# Path: src/vlm_engineering/service/routers/describe.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Image description and visual question answering."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..catalog import generation_model
from ..policy import reject_model_overrides, resolve_media_reference
from ..schemas import DescribeRequest, DescribeResponse
from .dependencies import service_config, service_runtime

router = APIRouter(prefix="/v1", tags=["inference"])


def _describe(payload: DescribeRequest, request: Request) -> DescribeResponse:
    config = service_config(request)
    runtime = service_runtime(request)
    reject_model_overrides(
        model_size=payload.model_size,
        model_id=payload.model_id,
        model_path=payload.model_path,
    )
    image = resolve_media_reference(payload.image, config)
    with runtime.limiter.slot("generation"):
        answer = runtime.generator.generate(
            [image],
            payload.prompt,
            max_new_tokens=payload.max_new_tokens,
            system_prompt=payload.system_prompt,
        )
    return DescribeResponse(answer=answer, model=generation_model(config))


@router.post("/describe", response_model=DescribeResponse)
def describe(payload: DescribeRequest, request: Request) -> DescribeResponse:
    """Describe an image or answer a question about it."""
    return _describe(payload, request)


@router.post("/analyze", response_model=DescribeResponse)
def analyze(payload: DescribeRequest, request: Request) -> DescribeResponse:
    """Alias of ``/v1/describe`` for structured visual analysis clients."""
    return _describe(payload, request)
