# Path: src/vlm_engineering/service/routers/models.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Configured model catalog; listing never loads weights."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ..catalog import embedding_model, generation_model, reranking_model
from ..config import ServiceConfig
from ..schemas import ModelInfo, ModelsResponse

router = APIRouter(prefix="/v1", tags=["models"])


def build_model_catalog(config: ServiceConfig) -> list[ModelInfo]:
    """Describe the models this process would serve without importing backends."""
    models = [
        ModelInfo(
            capability="generation",
            model=generation_model(config),
            provider=config.vlm_provider,
        )
    ]
    embedding = embedding_model(config)
    if embedding is not None:
        models.append(ModelInfo(capability="embedding", model=embedding, provider="qwen"))
    reranking = reranking_model(config)
    if reranking is not None:
        models.append(ModelInfo(capability="reranking", model=reranking, provider="qwen"))
    return models


@router.get("/models", response_model=ModelsResponse)
def list_models(request: Request) -> ModelsResponse:
    """List the configured generation, embedding and reranking models."""
    config: ServiceConfig = request.app.state.config
    return ModelsResponse(provider=config.vlm_provider, models=build_model_catalog(config))
