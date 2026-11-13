<!--
Path: CHANGELOG.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Changelog

## 0.4.0 — 2026-11-13: HTTP service API and retrieval platform

This release covers everything merged after 0.3.2: the v0.4 retrieval
platform, which had not been published yet, plus the new HTTP service API.

### HTTP service API (`[service]` extra)

- Add `vlm_engineering.service` with `create_app()`, a guarded import that
  explains how to install the extra, and `uvicorn
  vlm_engineering.service.app:app` support. `/health` serves without any
  model backend installed.
- Add synchronous routes: `GET /v1/models`, `POST /v1/describe`,
  `POST /v1/analyze`, `POST /v1/embed`, `POST /v1/rerank` and
  `POST /v1/query` (embed, search, rerank, answer).
- Add asynchronous job routes: `POST /v1/ingest`, `POST /v1/index`,
  `POST /v1/evaluate` return `202` with a UUID4 job id; `GET
  /v1/jobs/{id}` polls status/result/error and `DELETE /v1/jobs/{id}`
  removes finished jobs. The registry is bounded and TTL-based.
- Add environment configuration for every `VLM_SERVICE_*` setting:
  model selection, provider, request timeout, media/output roots, remote
  images (opt-in), store backend (`memory`, `sqlite-vec`, `qdrant`),
  Qdrant connection, concurrency, job retention and capacity.
- Reject per-request model overrides and confine local media/output
  references to their configured roots, including symlink escapes.
- Add per-capability admission control: saturated capabilities answer
  `503 service_busy` with `Retry-After` instead of queueing without bound.
- Normalize every failure into one `ErrorResponse` shape (400/404/409/
  422/429/500/503) without leaking tracebacks or backend internals.

### Retrieval and ingestion platform

- Add interchangeable vector stores (`memory`, `sqlite-vec`, `qdrant`)
  behind a single `VectorStore` protocol with upsert/delete/filter
  support.
- Add the ingestion pipeline (images and pre-rendered pages to
  traceable `VisualChunk` objects), a pluggable OCR interface and the
  `vlm-lab ingest`/`vlm-lab index` commands.
- Add the evaluation runner with Recall@K, nDCG@K, MRR and keypoint
  coverage, plus JSONL/JSON dataset loading and `vlm-lab evaluate`.
- Add prompt templates, the embedding/reranking response cache, typed
  runtime YAML profiles and actionable optional-dependency errors.
- Add OpenAI-compatible HTTP providers for generation, embedding and
  reranking alongside the Qwen3-VL providers.

## 0.3.2 — 2026-11-06: Reliability and security fixes

- Fix `TextChunker` so every emitted chunk respects `max_chars`, including an
  oversized paragraph after a non-empty chunk, and recognize CRLF paragraph
  boundaries; add regression tests for the budget contract.
- Raise Pillow to `>=13,<14` to clear 39 known CVEs flagged by `pip-audit`.
- Run CI on `feature/**` and `docs/**` branches, matching the repository's
  branch naming.
- Ship a PEP 561 `py.typed` marker and package it in wheels/sdists.
- Guard `python -m vlm_engineering` behind `__main__` and add subprocess tests.
- Replace the duplicate interactive-architecture page with a redirect to the
  Pages landing page; remove the tracked editor swap file.
- Add `make bootstrap` for a reproducible local virtual environment.

## 0.3.1 — Released: Multimodal retrieval / RAG

- Harden Qwen embedding/reranking adapters; preserve old names as aliases.
- Add 2B/8B/custom/local selection, batch size 1, dimensions, output validation,
  explicit raw/sigmoid scores, stable ranks, and unload support.
- Add strict local-image input contract and structural snapshot preflight.
- Add multimodal RAG with atomic re-indexing and all selected original images;
  fix stale state after empty re-indexing in the legacy visual RAG pipeline.
- Add bounded multi-image generation, local image preflight, inference mode,
  and a generation auto class compatible with Transformers 4.57.
- Add recognized operational error guidance without swallowing unknown errors.
- Add typed versioned runtime YAML and embed/rerank/validate-config CLI commands.
- Add qwen-retrieval/config extras; require Transformers v5 for the ST reranker.
- Add deterministic boundary tests and opt-in real-model embedding/reranker tests.
- Keep normal CI lightweight; add manual optional-runtime compatibility checks.
- Add Docker exclusions/config mount and correct incomplete contributor guidance.
- Compatibility: bare retrieval image paths now require explicit image objects;
  local corrupt images, unsupported modalities, malformed vectors and scores fail.

## Historical main changes — Configurable Qwen3-VL model selection

- Added first-class `2b`, `4b`, and `8b` Qwen3-VL Instruct presets.
- Kept `Qwen/Qwen3-VL-2B-Instruct` as the default model.
- Added arbitrary compatible Hugging Face `--model-id` support and explicit local/offline `--model-path`.
- Added `vlm-lab models` plus preset-aware model downloads.
- Added model-selection tests and user guidance for sizing, memory, offline loading and troubleshooting.
- Added a nine-scenario application cookbook covering captioning, VQA, diagram JSON, RAG chunking, text-only visual RAG, multimodal retrieval, model comparison, offline loading and batch analysis.

## Historical refactor notes (previously labeled 1.0.0)

The audited main package metadata was 0.2.1; this historical heading did not
match the package version and is not a claim that 1.0.0 was published.

### Production-oriented refactor

- Repositioned the repository as Vision-Language Engineering Lab.
- Added reusable CLIP encoder APIs.
- Added Qwen3-VL-2B-Instruct Hub/cache and explicit local/offline loading.
- Added explicit model downloader and secure `trust_remote_code` default.
- Added structured visual-document analysis, native+visual fusion and RAG-ready chunks.
- Added text-only visual RAG plus Qwen multimodal embedding/reranking adapters.
- Added CLI, examples, tests, Docker, CI/security workflows and bilingual teaching PDFs.
