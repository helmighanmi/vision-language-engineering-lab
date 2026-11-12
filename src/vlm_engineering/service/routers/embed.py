# Path: src/vlm_engineering/service/routers/embed.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Text, image and mixed modality embeddings."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Request

from ...exceptions import InputValidationError
from ...retrieval.model_options import validate_dimensions
from ..catalog import embedding_model
from ..policy import reject_model_overrides, resolve_media_reference
from ..schemas import EmbedRequest, EmbedResponse
from .dependencies import service_config, service_runtime

router = APIRouter(prefix="/v1", tags=["inference"])


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


@router.post("/embed", response_model=EmbedResponse)
def embed(payload: EmbedRequest, request: Request) -> EmbedResponse:
    """Embed text, an image, or both into a single vector."""
    config = service_config(request)
    runtime = service_runtime(request)
    reject_model_overrides(
        model_size=payload.model_size,
        model_id=payload.model_id,
        model_path=payload.model_path,
    )
    model = embedding_model(config)
    if model is None:
        raise InputValidationError(
            "Embedding is unavailable for this model configuration; "
            "use VLM_SERVICE_MODEL_SIZE=2b or 8b."
        )

    item: dict[str, str] = {}
    if payload.text is not None:
        item["text"] = payload.text
    if payload.image is not None:
        item["image"] = resolve_media_reference(payload.image, config)

    with runtime.limiter.slot("embedding"):
        vectors = np.asarray(runtime.embedder.encode([item], prompt=payload.prompt), dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) != 1:
        raise InputValidationError("Embedding provider returned an unexpected shape.")
    vector = vectors[0]

    if payload.dimensions is not None:
        validate_dimensions(payload.dimensions, len(vector))
        vector = vector[: payload.dimensions]
    if payload.normalize_embeddings:
        vector = _normalize(vector)

    return EmbedResponse(
        vector=[float(value) for value in vector],
        shape=list(vector.shape),
        model=model,
    )
