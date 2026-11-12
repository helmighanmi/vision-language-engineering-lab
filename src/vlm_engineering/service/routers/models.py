# Path: src/vlm_engineering/service/routers/models.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Configured model catalog; listing never loads weights."""

from __future__ import annotations

from fastapi import APIRouter, Request

from ...qwen.registry import resolve_qwen_model_id
from ..config import ServiceConfig
from ..schemas import ModelInfo, ModelsResponse

router = APIRouter(prefix="/v1", tags=["models"])

_RETRIEVAL_SIZES = frozenset({"2b", "8b"})


def _generation_model(config: ServiceConfig) -> str:
    if config.vlm_provider == "openai-compatible":
        return str(config.vlm_model)
    if config.model_path is not None:
        return str(config.model_path)
    if config.model_id is not None:
        return config.model_id
    return resolve_qwen_model_id(model_size=config.model_size)


def build_model_catalog(config: ServiceConfig) -> list[ModelInfo]:
    """Describe the models this process would serve without importing backends."""
    models = [
        ModelInfo(
            capability="generation",
            model=_generation_model(config),
            provider=config.vlm_provider,
        )
    ]
    if config.model_path is not None or config.model_id is not None:
        # The single selector also feeds the Qwen retrieval providers.
        source = str(config.model_path or config.model_id)
        models.append(ModelInfo(capability="embedding", model=source, provider="qwen"))
        models.append(ModelInfo(capability="reranking", model=source, provider="qwen"))
        return models
    if config.model_size is not None and config.model_size not in _RETRIEVAL_SIZES:
        # 4b is generation-only; the retrieval wrappers reject it.
        return models
    size = config.model_size or "2b"
    models.append(
        ModelInfo(
            capability="embedding",
            model=f"Qwen/Qwen3-VL-Embedding-{size.upper()}",
            provider="qwen",
        )
    )
    models.append(
        ModelInfo(
            capability="reranking",
            model=f"Qwen/Qwen3-VL-Reranker-{size.upper()}",
            provider="qwen",
        )
    )
    return models


@router.get("/models", response_model=ModelsResponse)
def list_models(request: Request) -> ModelsResponse:
    """List the configured generation, embedding and reranking models."""
    config: ServiceConfig = request.app.state.config
    return ModelsResponse(provider=config.vlm_provider, models=build_model_catalog(config))
