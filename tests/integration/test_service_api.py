# Path: tests/integration/test_service_api.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Integration tests for the service HTTP API (no model backends needed)."""

from __future__ import annotations

import asyncio
import importlib.metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vlm_engineering.exceptions import (
    InputValidationError,
    ModelLoadError,
    OptionalDependencyError,
    VLMEngineeringError,
)
from vlm_engineering.retrieval.in_memory import SearchResult
from vlm_engineering.service.concurrency import ServiceBusyError

pytest.importorskip("fastapi", reason="the service extra is not installed")

from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from vlm_engineering.service import create_app  # noqa: E402
from vlm_engineering.service.app import _validation_error_response  # noqa: E402
from vlm_engineering.service.config import ServiceConfig  # noqa: E402
from vlm_engineering.service.routers.query import NO_EVIDENCE_ANSWER  # noqa: E402
from vlm_engineering.service.runtime import ServiceRuntime  # noqa: E402
from vlm_engineering.service.schemas import package_version  # noqa: E402


@pytest.fixture(name="client")
def client_fixture() -> TestClient:
    return TestClient(create_app())


def test_health_reports_ok_and_package_version(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "version": package_version(),
    }
    assert package_version() == importlib.metadata.version(
        "vision-language-engineering-lab"
    )


def test_ready_reports_ok(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_openapi_schema_lists_public_routes(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {
        "/health",
        "/ready",
        "/v1/models",
        "/v1/describe",
        "/v1/analyze",
        "/v1/embed",
        "/v1/rerank",
        "/v1/query",
        "/v1/ingest",
        "/v1/index",
        "/v1/evaluate",
        "/v1/jobs/{job_id}",
    }


def test_unknown_route_is_standard_404(client: TestClient) -> None:
    assert client.get("/v1/not-a-route").status_code == 404


def test_models_catalog_lists_configured_capabilities(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "qwen"
    assert [item["capability"] for item in body["models"]] == [
        "generation",
        "embedding",
        "reranking",
    ]
    assert body["models"][0]["model"].endswith("Qwen3-VL-2B-Instruct")


def test_validation_error_response_shape() -> None:
    """The 422 handler maps FastAPI issues onto ErrorResponse fields."""
    exc = RequestValidationError(
        errors=[
            {"loc": ("body", "image"), "msg": "field required"},
            {"loc": ("body", "top_k"), "msg": "must be positive"},
        ],
        body=None,
    )
    response = asyncio.run(_validation_error_response(None, exc))  # type: ignore[arg-type]
    assert response.status_code == 422
    body = response.body  # type: ignore[attr-defined]
    assert b"validation_error" in body
    assert b"body.image: field required" in body
    assert b"body.top_k: must be positive" in body


def test_validation_error_response_with_no_issues() -> None:
    exc = RequestValidationError(errors=[], body=None)
    response = asyncio.run(_validation_error_response(None, exc))  # type: ignore[arg-type]
    assert response.status_code == 422


# -- Runtime wiring ----------------------------------------------------------

def test_app_state_exposes_config_and_runtime() -> None:
    app = create_app(ServiceConfig(max_concurrency=3))
    assert app.state.config.max_concurrency == 3
    assert isinstance(app.state.runtime, ServiceRuntime)
    assert app.state.runtime.config is app.state.config
    assert app.state.runtime.loaded_backends() == {
        "generator": False,
        "embedder": False,
        "reranker": False,
    }


def test_create_app_defaults_to_environment_config() -> None:
    app = create_app()
    assert isinstance(app.state.config, ServiceConfig)


# -- Error mapping -----------------------------------------------------------

@pytest.mark.parametrize(
    ("error", "status_code", "code"),
    [
        (InputValidationError("bad input"), 400, "invalid_request"),
        (ValueError("bad value"), 400, "invalid_request"),
        (FileNotFoundError("missing"), 400, "file_not_found"),
        (PermissionError("denied"), 400, "permission_denied"),
        (OptionalDependencyError("install it"), 503, "dependency_unavailable"),
        (ImportError("no module"), 503, "dependency_unavailable"),
        (ModelLoadError("cannot load"), 503, "model_unavailable"),
        (ServiceBusyError("generation"), 503, "service_busy"),
        (VLMEngineeringError("engine"), 400, "request_error"),
        (RuntimeError("secret-token"), 500, "internal_error"),
    ],
)
def test_exceptions_map_to_error_response(
    error: Exception, status_code: int, code: str
) -> None:
    app = create_app(ServiceConfig())

    @app.get("/boom")
    def boom() -> None:
        raise error

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom")
    assert response.status_code == status_code
    body = response.json()
    assert body["error"] == code
    if status_code == 500:
        assert body["detail"] is None
        assert "secret-token" not in response.text
    else:
        assert body["detail"] == str(error)


def test_busy_error_advertises_retry_after() -> None:
    app = create_app(ServiceConfig())

    @app.get("/busy")
    def busy() -> None:
        raise ServiceBusyError("generation")

    response = TestClient(app, raise_server_exceptions=False).get("/busy")
    assert response.status_code == 503
    assert response.json()["error"] == "service_busy"
    assert response.headers["Retry-After"] == "1"


# -- Inference routes --------------------------------------------------------

class _FakeGenerator:
    def __init__(self) -> None:
        self.generate_calls: list[dict[str, object]] = []
        self.text_calls: list[dict[str, object]] = []

    def generate(
        self,
        images: list[str],
        prompt: str,
        *,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        self.generate_calls.append(
            {
                "images": list(images),
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "system_prompt": system_prompt,
            }
        )
        return "generated description"

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
    ) -> str:
        self.text_calls.append({"prompt": prompt, "max_new_tokens": max_new_tokens})
        return "generated answer"


class _FakeEmbedder:
    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions
        self.inputs: list[list[dict[str, str]]] = []

    def encode(
        self, inputs: list[dict[str, str]], *, prompt: str | None = None
    ) -> np.ndarray:
        self.inputs.append(inputs)
        return np.ones((len(inputs), self.dimensions), dtype=np.float64)


class _FakeReranker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def rerank(
        self,
        query: object,
        documents: list[object],
        *,
        top_k: int | None = None,
        prompt: str | None = None,
    ) -> list[SearchResult]:
        self.calls.append({"query": query, "documents": documents, "top_k": top_k})
        scores = [-2.5, 0.5, 0.25]
        ordered = [
            SearchResult(item=document, score=scores[index], rank=index + 1)
            for index, document in enumerate(documents)
        ]
        return ordered[:top_k] if top_k is not None else ordered


class _FakeStore:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.searches: list[dict[str, object]] = []
        self.added: list[dict[str, object]] = []
        self.persisted: list[Path] = []

    def add(
        self,
        embeddings: np.ndarray,
        items: list[object],
        *,
        ids: list[str] | None = None,
    ) -> None:
        self.added.append({"embeddings": embeddings, "items": items, "ids": ids})

    def persist(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("persisted", encoding="utf-8")
        self.persisted.append(destination)

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
        filter: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        self.searches.append({"vector": query_embedding, "top_k": top_k})
        return self.results[:top_k]


class _FakeAnalysisGenerator:
    """Returns a valid structured analysis for pipeline ingestion."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, image: str, prompt: str, *, max_new_tokens: int = 512) -> str:
        self.calls.append(image)
        return (
            '{"page_type":"diagram","title":"Architecture","summary":"A diagram.",'
            '"entities":["encoder"],"relations":[],"important_text":[],"uncertainties":[]}'
        )


class _Harness:
    def __init__(
        self,
        tmp_path: Path,
        *,
        results: list[SearchResult] | None = None,
        generator: Any | None = None,
        **config: object,
    ) -> None:
        self.media = tmp_path / "media"
        self.media.mkdir()
        self.output = tmp_path / "output"
        self.output.mkdir()
        self.image = self.media / "page.png"
        self.image.write_bytes(b"image-bytes")
        self.config = ServiceConfig(
            media_root=self.media,
            output_root=self.output,
            **config,  # type: ignore[arg-type]
        )
        self.generator = generator if generator is not None else _FakeGenerator()
        self.embedder = _FakeEmbedder()
        self.reranker = _FakeReranker()
        self.store = _FakeStore(
            results
            if results is not None
            else [
                SearchResult({"text": "evidence one"}, 0.9, 1),
                SearchResult({"text": "evidence two"}, 0.8, 2),
                SearchResult({"text": "evidence three"}, 0.7, 3),
            ]
        )
        self.runtime = ServiceRuntime(
            self.config,
            generator=self.generator,
            embedder=self.embedder,
            reranker=self.reranker,
            stores={"memory": self.store},
        )
        self.app = create_app(self.config)
        self.app.state.runtime = self.runtime
        self.client = TestClient(self.app)


def test_describe_resolves_media_and_generates(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post(
        "/v1/describe",
        json={"image": str(harness.image), "prompt": "Describe.", "max_new_tokens": 32},
    )
    assert response.status_code == 200, response.text
    assert response.json()["answer"] == "generated description"
    assert harness.generator.generate_calls[0]["images"] == [str(harness.image.resolve())]
    assert harness.generator.generate_calls[0]["max_new_tokens"] == 32


def test_analyze_is_an_alias_of_describe(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post("/v1/analyze", json={"image": str(harness.image)})
    assert response.status_code == 200
    assert harness.generator.generate_calls


def test_describe_rejects_paths_outside_media_root(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"x")
    response = harness.client.post("/v1/describe", json={"image": str(outside)})
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"
    assert not harness.generator.generate_calls


def test_describe_rejects_model_overrides(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post(
        "/v1/describe", json={"image": str(harness.image), "model_size": "8b"}
    )
    assert response.status_code == 400
    assert "Per-request model selection" in response.json()["detail"]


def test_embed_text_and_image_and_dimensions(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post(
        "/v1/embed",
        json={"text": "hello", "image": str(harness.image), "dimensions": 64},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["shape"] == [64]
    assert len(body["vector"]) == 64
    assert body["model"].endswith("Qwen3-VL-Embedding-2B")
    sent = harness.embedder.inputs[0][0]
    assert sent == {"text": "hello", "image": str(harness.image.resolve())}
    assert np.isclose(np.linalg.norm(body["vector"]), 1.0)


def test_embed_requires_text_or_image(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post("/v1/embed", json={})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_embed_rejects_invalid_dimensions(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post("/v1/embed", json={"text": "hello", "dimensions": 8})
    assert response.status_code == 400


def test_rerank_returns_ranked_results_and_normalizes(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post(
        "/v1/rerank",
        json={
            "query": "which?",
            "documents": [{"text": "a"}, {"text": "b"}],
            "normalize_scores": True,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["rank"] for item in body["results"]] == [1, 2]
    assert 0.0 < body["results"][0]["score"] < 1.0
    assert body["model"].endswith("Qwen3-VL-Reranker-2B")


def test_rerank_requires_documents(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post("/v1/rerank", json={"query": "q", "documents": []})
    assert response.status_code == 422


def test_query_retrieves_reranks_and_answers(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post("/v1/query", json={"question": "What is shown?", "top_k": 2})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["answer"] == "generated answer"
    assert len(body["chunks"]) == 2
    assert harness.store.searches[0]["top_k"] == 12
    assert "SOURCE 1: evidence one" in str(harness.generator.text_calls[0]["prompt"])


def test_query_with_image_sends_it_to_the_generator(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post(
        "/v1/query", json={"question": "Compare.", "query_image": str(harness.image)}
    )
    assert response.status_code == 200, response.text
    assert harness.generator.generate_calls[0]["images"] == [str(harness.image.resolve())]


def test_query_without_evidence_does_not_call_the_generator(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, results=[])
    response = harness.client.post("/v1/query", json={"question": "Anything?"})
    assert response.status_code == 200
    assert response.json()["answer"] == NO_EVIDENCE_ANSWER
    assert response.json()["chunks"] == []
    assert not harness.generator.text_calls


def test_query_rejects_candidate_k_below_top_k(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post(
        "/v1/query", json={"question": "q", "top_k": 5, "candidate_k": 2}
    )
    assert response.status_code == 400


def test_saturated_capability_returns_503_with_retry_after(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, max_concurrency=1)
    with harness.runtime.limiter.slot("generation"):
        response = harness.client.post("/v1/describe", json={"image": str(harness.image)})
    assert response.status_code == 503
    assert response.json()["error"] == "service_busy"
    assert response.headers["Retry-After"] == "1"
    assert not harness.generator.generate_calls


# -- Async jobs ---------------------------------------------------------------

def _wait_for_job(client: TestClient, job_id: str) -> dict[str, Any]:
    response = client.get(f"/v1/jobs/{job_id}")
    assert response.status_code == 200, response.text
    return response.json()


def test_ingest_job_reaches_completed_with_chunks(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post(
        "/v1/ingest",
        json={"images": [str(harness.image)], "document_id": "doc-1", "no_vlm": True},
    )
    assert response.status_code == 202, response.text
    body = response.json()
    assert body["status"] == "accepted"

    status = _wait_for_job(harness.client, body["job_id"])
    assert status["status"] == "completed"
    result = status["result"]
    assert result["total_chunks"] == 1
    assert result["results"][0]["document_id"] == "doc-1"
    assert result["results"][0]["image_ref"] == str(harness.image.resolve())


def test_ingest_expands_directories_and_ignores_other_files(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    nested = harness.media / "pages"
    nested.mkdir()
    (nested / "b.png").write_bytes(b"b")
    (nested / "a.jpg").write_bytes(b"a")
    (nested / "notes.txt").write_text("ignored", encoding="utf-8")

    response = harness.client.post(
        "/v1/ingest", json={"images": [str(nested)], "no_vlm": True}
    )
    assert response.status_code == 202, response.text
    status = _wait_for_job(harness.client, response.json()["job_id"])
    assert status["status"] == "completed"
    assert status["result"]["total_chunks"] == 2


def test_ingest_uses_the_configured_generator(tmp_path: Path) -> None:
    generator = _FakeAnalysisGenerator()
    harness = _Harness(tmp_path, generator=generator)
    response = harness.client.post("/v1/ingest", json={"images": [str(harness.image)]})
    assert response.status_code == 202, response.text
    status = _wait_for_job(harness.client, response.json()["job_id"])
    assert status["status"] == "completed"
    assert generator.calls == [str(harness.image)]
    assert status["result"]["results"][0]["title"] == "Architecture"


def test_ingest_rejects_missing_images(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post(
        "/v1/ingest", json={"images": [str(harness.media / "missing.png")]}
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_index_job_persists_into_the_output_root(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post(
        "/v1/index",
        json={
            "chunks": [
                {"chunk_id": "c1", "text": "alpha"},
                {"chunk_id": "c2", "text": "beta"},
            ],
            "store": "memory",
            "output": "index/main",
        },
    )
    assert response.status_code == 202, response.text
    status = _wait_for_job(harness.client, response.json()["job_id"])
    assert status["status"] == "completed"
    assert status["result"] == {
        "indexed": 2,
        "store": "memory",
        "output": "index/main",
    }
    assert harness.store.added[0]["ids"] == ["c1", "c2"]
    assert harness.store.persisted[0] == (harness.output / "index" / "main").resolve()


def test_index_rejects_output_outside_the_root(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post(
        "/v1/index",
        json={"chunks": [{"text": "alpha"}], "output": "../escape"},
    )
    assert response.status_code == 400
    assert "'..'" in response.json()["detail"]


def test_index_rejects_chunks_without_content(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post("/v1/index", json={"chunks": [{"chunk_id": "c1"}]})
    assert response.status_code == 400
    assert "text or an image" in response.json()["detail"]


def test_index_rejects_an_explicit_model_size_override(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post(
        "/v1/index",
        json={"chunks": [{"text": "alpha"}], "embed_model_size": "8b"},
    )
    assert response.status_code == 400
    assert "Per-request model selection" in response.json()["detail"]


def test_index_accepts_the_configured_model_size(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, model_size="8b")
    response = harness.client.post(
        "/v1/index",
        json={"chunks": [{"text": "alpha"}], "embed_model_size": "8b"},
    )
    assert response.status_code == 202, response.text
    assert _wait_for_job(harness.client, response.json()["job_id"])["status"] == "completed"


def test_index_rejects_a_model_path_override(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post(
        "/v1/index",
        json={"chunks": [{"text": "alpha"}], "embed_model_path": "/models/embed"},
    )
    assert response.status_code == 400
    assert "Per-request model selection" in response.json()["detail"]


def test_evaluate_job_reports_cli_compatible_metrics(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post(
        "/v1/evaluate",
        json={
            "cases": [
                {"case_id": "case-1", "query": "alpha", "relevant_ids": ["c1"]},
            ],
            "chunks": [
                {
                    "chunk_id": "c1",
                    "document_id": "doc",
                    "source_file": "doc.pdf",
                    "page": 1,
                    "chunk_type": "text",
                    "title": "Page 1",
                    "text": "alpha",
                }
            ],
            "k_values": [1],
        },
    )
    assert response.status_code == 202, response.text
    status = _wait_for_job(harness.client, response.json()["job_id"])
    assert status["status"] == "completed"
    result = status["result"]
    assert result["case_count"] == 1
    assert result["k_values"] == [1]
    assert result["aggregate"]["recall@1"] == 1.0
    assert result["aggregate"]["mrr@1"] == 1.0
    assert result["latency_ms_mean"] >= 0.0


def test_evaluate_rejects_invalid_k_values(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post(
        "/v1/evaluate",
        json={"cases": [{"case_id": "c", "query": "q"}], "chunks": [{"text": "a"}], "k_values": [0]},
    )
    assert response.status_code == 422


def test_evaluate_rejects_invalid_cases(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.post(
        "/v1/evaluate",
        json={"cases": [{"query": "q"}], "chunks": [{"text": "a"}]},
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_request"


def test_unknown_job_returns_404(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    response = harness.client.get("/v1/jobs/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"] == "not_found"


def test_delete_finished_job_then_poll_returns_404(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    submitted = harness.client.post(
        "/v1/ingest", json={"images": [str(harness.image)], "no_vlm": True}
    )
    job_id = submitted.json()["job_id"]
    _wait_for_job(harness.client, job_id)

    deleted = harness.client.delete(f"/v1/jobs/{job_id}")
    assert deleted.status_code == 204
    assert harness.client.get(f"/v1/jobs/{job_id}").status_code == 404


def test_deleting_an_active_job_is_a_conflict(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    job = harness.app.state.jobs.create("ingest")
    response = harness.client.delete(f"/v1/jobs/{job.job_id}")
    assert response.status_code == 409
    assert response.json()["error"] == "job_conflict"


def test_job_capacity_saturation_returns_503(tmp_path: Path) -> None:
    harness = _Harness(tmp_path, max_jobs=1)
    first = harness.app.state.jobs.create("index")
    response = harness.client.post("/v1/ingest", json={"images": [str(harness.image)]})
    assert response.status_code == 503
    assert response.json()["error"] == "service_busy"
    assert harness.app.state.jobs.get(first.job_id).status == "accepted"


def test_lifespan_closes_the_runtime_on_shutdown() -> None:
    class _FakeProvider:
        def __init__(self) -> None:
            self.unload_calls = 0

        def unload(self) -> None:
            self.unload_calls += 1

    fake = _FakeProvider()
    app = create_app(ServiceConfig())
    app.state.runtime = ServiceRuntime(ServiceConfig(), generator=fake)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200

    assert fake.unload_calls == 1
