# Path: src/vlm_engineering/service/routers/extract.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""NuExtract3 document extraction over the configured VLM generator."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ...extraction import NuExtract3Extractor
from ..catalog import generation_model
from ..policy import resolve_media_reference
from ..schemas import ExtractRequest, ExtractResponse
from .dependencies import service_config, service_runtime

router = APIRouter(prefix="/v1", tags=["inference"])


def _extract(payload: ExtractRequest, request: Request) -> ExtractResponse:
    config = service_config(request)
    runtime = service_runtime(request)
    image = resolve_media_reference(payload.image, config)
    extractor = NuExtract3Extractor(
        runtime.generator, max_new_tokens=payload.max_new_tokens
    )

    with runtime.limiter.slot("generation"):
        if payload.markdown:
            markdown = extractor.extract_markdown(
                [image],
                instructions=payload.instructions,
                enable_thinking=payload.enable_thinking,
                max_new_tokens=payload.max_new_tokens,
            )
            return ExtractResponse(markdown=markdown, model=generation_model(config))

        data = extractor.extract(
            [image],
            payload.template or {},
            instructions=payload.instructions,
            enable_thinking=payload.enable_thinking,
            max_new_tokens=payload.max_new_tokens,
        )
    return ExtractResponse(data=data, model=generation_model(config))


@router.post("/extract", response_model=ExtractResponse)
def extract(payload: ExtractRequest, request: Request) -> ExtractResponse:
    """Extract template-guided JSON or full Markdown content from a document."""
    return _extract(payload, request)
