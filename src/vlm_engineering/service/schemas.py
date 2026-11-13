# Path: src/vlm_engineering/service/schemas.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Pydantic request/response schemas for the HTTP service."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_DISTRIBUTION = "vision-language-engineering-lab"


def package_version() -> str:
    """Return the installed package version, or a placeholder when unknown."""
    try:
        return version(_DISTRIBUTION)
    except PackageNotFoundError:  # pragma: no cover - only for uninstalled runs
        return "0.0.0+unknown"


class ErrorResponse(BaseModel):
    """Standard error response."""

    model_config = ConfigDict(from_attributes=True)

    error: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(from_attributes=True)

    status: str = "ok"
    version: str = Field(default_factory=package_version)


class StoreReadiness(BaseModel):
    """Result of probing the configured vector store."""

    model_config = ConfigDict(from_attributes=True)

    kind: str
    ready: bool
    detail: Optional[str] = None


class ReadinessResponse(BaseModel):
    """Readiness report; probing never runs inference."""

    model_config = ConfigDict(from_attributes=True)

    status: str = "ok"
    version: str = Field(default_factory=package_version)
    backends: dict[str, bool] = Field(default_factory=dict)
    store: StoreReadiness


# -- Models -----------------------------------------------------------------

class ModelInfo(BaseModel):
    """One capability/model pair served by this process."""

    model_config = ConfigDict(from_attributes=True)

    capability: str
    model: str
    provider: str


class ModelsResponse(BaseModel):
    """Configured model catalog; constructing it never loads weights."""

    model_config = ConfigDict(from_attributes=True)

    provider: str
    models: list[ModelInfo]


# -- Describe / Analyze -----------------------------------------------------

class DescribeRequest(BaseModel):
    """Request to describe or question an image."""

    model_config = ConfigDict(from_attributes=True)

    image: str = Field(..., description="Local image path or HTTP(S) URL.")
    prompt: str = Field(
        default="Describe this image precisely.", description="Generation prompt."
    )
    model_size: Optional[str] = Field(
        default=None, description="Qwen3-VL preset: 2b, 4b, or 8b."
    )
    model_id: Optional[str] = Field(
        default=None, description="Explicit Hugging Face model ID."
    )
    model_path: Optional[str] = Field(
        default=None, description="Explicit local model directory."
    )
    max_new_tokens: int = Field(default=512, gt=0, le=4096)
    system_prompt: Optional[str] = None

    @field_validator("image")
    @classmethod
    def _image_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("image must not be empty.")
        return value


class DescribeResponse(BaseModel):
    """Response from describe/analyze."""

    model_config = ConfigDict(from_attributes=True)

    answer: str
    model: str


# -- Embed ------------------------------------------------------------------

class EmbedRequest(BaseModel):
    """Request to embed text, image, or both."""

    model_config = ConfigDict(from_attributes=True)

    text: Optional[str] = None
    image: Optional[str] = None
    model_size: Optional[str] = Field(default=None, description="Embedding preset: 2b or 8b.")
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    dimensions: Optional[int] = None
    normalize_embeddings: bool = True
    prompt: Optional[str] = None

    @field_validator("text", "image")
    @classmethod
    def _not_empty(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("field must not be empty when provided.")
        return value

    @model_validator(mode="after")
    def _text_or_image(self) -> "EmbedRequest":
        if self.text is None and self.image is None:
            raise ValueError("provide text, image, or both.")
        return self


class EmbedResponse(BaseModel):
    """Response from embed."""

    model_config = ConfigDict(from_attributes=True)

    vector: list[float]
    shape: list[int]
    model: str


# -- Rerank -----------------------------------------------------------------

class RerankRequest(BaseModel):
    """Request to rerank candidates."""

    model_config = ConfigDict(from_attributes=True)

    query: str
    documents: list[dict[str, str]] = Field(..., min_length=1)
    model_size: Optional[str] = Field(default=None, description="Reranker preset: 2b or 8b.")
    model_id: Optional[str] = None
    model_path: Optional[str] = None
    top_k: Optional[int] = Field(default=None, gt=0)
    normalize_scores: bool = False
    prompt: Optional[str] = None

    @field_validator("query")
    @classmethod
    def _query_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty.")
        return value


class RerankResult(BaseModel):
    """One reranked result."""

    model_config = ConfigDict(from_attributes=True)

    item: dict[str, str]
    score: float
    rank: int


class RerankResponse(BaseModel):
    """Response from rerank."""

    model_config = ConfigDict(from_attributes=True)

    results: list[RerankResult]
    model: str


# -- Ingest -----------------------------------------------------------------

class IngestRequest(BaseModel):
    """Request to ingest images into chunks."""

    model_config = ConfigDict(from_attributes=True)

    images: list[str] = Field(..., min_length=1)
    document_id: str = "ingested"
    source_file: str = "images"
    ocr: bool = False
    no_vlm: bool = False

    @field_validator("images")
    @classmethod
    def _images_not_empty(cls, value: list[str]) -> list[str]:
        for item in value:
            if not item.strip():
                raise ValueError("image paths must not be empty.")
        return value


class IngestResultItem(BaseModel):
    """One ingested chunk."""

    model_config = ConfigDict(from_attributes=True)

    chunk_id: str
    document_id: str
    source_file: str
    page: int
    chunk_type: str
    title: str
    text: str
    image_ref: str
    errors: list[str] = []


class IngestResponse(BaseModel):
    """Response from ingest."""

    model_config = ConfigDict(from_attributes=True)

    results: list[IngestResultItem]
    total_chunks: int
    total_errors: int


# -- Index ------------------------------------------------------------------

class IndexRequest(BaseModel):
    """Request to index chunks into a vector store."""

    model_config = ConfigDict(from_attributes=True)

    chunks: list[dict[str, Any]] = Field(..., min_length=1)
    store: str = Field(
        default="memory", description="memory, sqlite-vec, or qdrant."
    )
    output: str = "output/index"
    embed_model_size: str = Field(default="2b", description="Embedding preset: 2b or 8b.")
    embed_model_path: Optional[str] = None

    @field_validator("store")
    @classmethod
    def _valid_store(cls, value: str) -> str:
        if value not in ("memory", "sqlite-vec", "qdrant"):
            raise ValueError("store must be memory, sqlite-vec, or qdrant.")
        return value


class IndexResponse(BaseModel):
    """Response from index."""

    model_config = ConfigDict(from_attributes=True)

    indexed: int
    store: str
    output: str


# -- Query / Answer ---------------------------------------------------------

class QueryRequest(BaseModel):
    """Request to retrieve and answer a question."""

    model_config = ConfigDict(from_attributes=True)

    question: str
    query_image: Optional[str] = None
    top_k: Optional[int] = Field(default=None, gt=0)
    candidate_k: Optional[int] = Field(default=None, gt=0)
    max_new_tokens: Optional[int] = Field(default=None, gt=0, le=4096)

    @field_validator("question")
    @classmethod
    def _question_not_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty.")
        return value


class QueryResponse(BaseModel):
    """Response from query/answer."""

    model_config = ConfigDict(from_attributes=True)

    answer: str
    chunks: list[dict[str, Any]]
    model: str


# -- Evaluate ---------------------------------------------------------------

class EvaluateRequest(BaseModel):
    """Request to run retrieval evaluation."""

    model_config = ConfigDict(from_attributes=True)

    cases: list[dict[str, Any]] = Field(..., min_length=1)
    chunks: list[dict[str, Any]] = Field(..., min_length=1)
    k_values: list[int] = Field(default=[1, 3, 5], min_length=1)
    prompt: Optional[str] = None

    @field_validator("k_values")
    @classmethod
    def _positive_k_values(cls, value: list[int]) -> list[int]:
        if any(item <= 0 for item in value):
            raise ValueError("k_values must contain positive integers.")
        return value


class EvaluateResponse(BaseModel):
    """Response from evaluate."""

    model_config = ConfigDict(from_attributes=True)

    aggregate: dict[str, float]
    case_count: int
    k_values: list[int]
    latency_ms_mean: float


# -- Jobs -------------------------------------------------------------------

class JobResponse(BaseModel):
    """Async job submission response."""

    model_config = ConfigDict(from_attributes=True)

    job_id: str
    status: str = "accepted"


class JobStatusResponse(BaseModel):
    """Async job status response."""

    model_config = ConfigDict(from_attributes=True)

    job_id: str
    status: str  # accepted, running, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
