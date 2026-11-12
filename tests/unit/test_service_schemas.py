# Path: tests/unit/test_service_schemas.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Unit tests for the HTTP service request/response schemas."""

from __future__ import annotations

import importlib.metadata

import pytest
from pydantic import ValidationError

pytest.importorskip("pydantic", reason="the service extra is not installed")

from vlm_engineering.service.schemas import (  # noqa: E402
    DescribeRequest,
    DescribeResponse,
    EmbedRequest,
    EmbedResponse,
    ErrorResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    IngestRequest,
    IngestResponse,
    IngestResultItem,
    JobResponse,
    JobStatusResponse,
    ModelInfo,
    ModelsResponse,
    QueryRequest,
    QueryResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
    package_version,
)

# -- Version -----------------------------------------------------------------

def test_package_version_matches_installed_distribution() -> None:
    assert package_version() == importlib.metadata.version(
        "vision-language-engineering-lab"
    )


def test_health_response_defaults_to_package_version() -> None:
    assert HealthResponse().version == package_version()
    assert HealthResponse().status == "ok"


# -- ErrorResponse -----------------------------------------------------------

def test_error_response_optional_detail() -> None:
    assert ErrorResponse(error="boom").detail is None
    assert ErrorResponse(error="boom", detail="why").detail == "why"


# -- Models ------------------------------------------------------------------

def test_models_response_shape() -> None:
    response = ModelsResponse(
        provider="qwen",
        models=[ModelInfo(capability="generation", model="m", provider="qwen")],
    )
    assert response.models[0].capability == "generation"
    assert response.models[0].provider == "qwen"


# -- Describe ----------------------------------------------------------------

def test_describe_request_defaults() -> None:
    request = DescribeRequest(image="img.png")
    assert request.prompt == "Describe this image precisely."
    assert request.max_new_tokens == 512
    assert request.model_size is None
    assert request.system_prompt is None


def test_describe_request_rejects_blank_image() -> None:
    with pytest.raises(ValidationError):
        DescribeRequest(image="   ")


@pytest.mark.parametrize("tokens", [0, -1, 4097])
def test_describe_request_rejects_bad_max_new_tokens(tokens: int) -> None:
    with pytest.raises(ValidationError):
        DescribeRequest(image="img.png", max_new_tokens=tokens)


def test_describe_response_shape() -> None:
    response = DescribeResponse(answer="a cat", model="2b")
    assert response.answer == "a cat"
    assert response.model == "2b"


# -- Embed -------------------------------------------------------------------

def test_embed_request_accepts_text_only_and_image_only() -> None:
    assert EmbedRequest(text="hello").image is None
    assert EmbedRequest(image="img.png").text is None


def test_embed_request_rejects_blank_fields() -> None:
    with pytest.raises(ValidationError):
        EmbedRequest(text="   ")
    with pytest.raises(ValidationError):
        EmbedRequest(image="")


def test_embed_request_normalization_flag_defaults_true() -> None:
    assert EmbedRequest(text="hello").normalize_embeddings is True


def test_embed_response_shape() -> None:
    response = EmbedResponse(vector=[0.1, 0.2], shape=[1, 2], model="2b")
    assert response.shape == [1, 2]


# -- Rerank ------------------------------------------------------------------

def test_rerank_request_defaults() -> None:
    request = RerankRequest(query="q", documents=[{"text": "d"}])
    assert request.top_k is None
    assert request.normalize_scores is False


def test_rerank_request_rejects_bad_top_k() -> None:
    with pytest.raises(ValidationError):
        RerankRequest(query="q", documents=[{"text": "d"}], top_k=0)


def test_rerank_result_and_response_shape() -> None:
    result = RerankResult(item={"text": "d"}, score=0.5, rank=1)
    response = RerankResponse(results=[result], model="2b")
    assert response.results[0].rank == 1


# -- Ingest ------------------------------------------------------------------

def test_ingest_request_defaults() -> None:
    request = IngestRequest(images=["page.png"])
    assert request.document_id == "ingested"
    assert request.source_file == "images"
    assert request.ocr is False
    assert request.no_vlm is False


def test_ingest_request_requires_at_least_one_image() -> None:
    with pytest.raises(ValidationError):
        IngestRequest(images=[])


def test_ingest_request_rejects_blank_image_paths() -> None:
    with pytest.raises(ValidationError):
        IngestRequest(images=["page.png", "  "])


def test_ingest_result_item_errors_default_empty() -> None:
    item = IngestResultItem(
        chunk_id="c1",
        document_id="doc",
        source_file="s",
        page=1,
        chunk_type="text",
        title="t",
        text="x",
        image_ref="r",
    )
    assert item.errors == []


def test_ingest_response_counts() -> None:
    response = IngestResponse(results=[], total_chunks=0, total_errors=0)
    assert response.total_chunks == 0


# -- Index -------------------------------------------------------------------

def test_index_request_defaults() -> None:
    request = IndexRequest(chunks=[{"chunk_id": "c1", "text": "x"}])
    assert request.store == "memory"
    assert request.output == "output/index"
    assert request.embed_model_size == "2b"


@pytest.mark.parametrize("store", ["redis", "", "SQLITE-VEC"])
def test_index_request_rejects_unknown_store(store: str) -> None:
    with pytest.raises(ValidationError):
        IndexRequest(chunks=[{"chunk_id": "c1"}], store=store)


def test_index_request_requires_chunks() -> None:
    with pytest.raises(ValidationError):
        IndexRequest(chunks=[])


def test_index_response_shape() -> None:
    response = IndexResponse(indexed=3, store="memory", output="out")
    assert response.indexed == 3


# -- Query -------------------------------------------------------------------

def test_query_request_defaults() -> None:
    request = QueryRequest(question="What is shown?")
    assert request.top_k is None
    assert request.candidate_k is None
    assert request.query_image is None


def test_query_request_rejects_blank_question() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="  ")


@pytest.mark.parametrize("value", [0, -3])
def test_query_request_rejects_bad_top_k(value: int) -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="q", top_k=value)


def test_query_request_rejects_bad_max_new_tokens() -> None:
    with pytest.raises(ValidationError):
        QueryRequest(question="q", max_new_tokens=4097)


def test_query_response_shape() -> None:
    response = QueryResponse(answer="a", chunks=[], model="2b")
    assert response.chunks == []


# -- Evaluate ----------------------------------------------------------------

def test_evaluate_request_defaults() -> None:
    request = EvaluateRequest(cases=[{"id": "c"}], chunks=[{"chunk_id": "c"}])
    assert request.k_values == [1, 3, 5]
    assert request.prompt is None


def test_evaluate_request_requires_cases_and_chunks() -> None:
    with pytest.raises(ValidationError):
        EvaluateRequest(cases=[], chunks=[{"chunk_id": "c"}])
    with pytest.raises(ValidationError):
        EvaluateRequest(cases=[{"id": "c"}], chunks=[])


def test_evaluate_response_shape() -> None:
    response = EvaluateResponse(
        aggregate={"recall@1": 1.0},
        case_count=1,
        k_values=[1],
        latency_ms_mean=1.5,
    )
    assert response.case_count == 1


# -- Jobs --------------------------------------------------------------------

def test_job_response_defaults_to_accepted() -> None:
    assert JobResponse(job_id="abc").status == "accepted"


def test_job_status_response_allows_pending_result() -> None:
    status = JobStatusResponse(job_id="abc", status="running")
    assert status.result is None
    assert status.error is None
