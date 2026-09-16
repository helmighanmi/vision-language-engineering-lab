<!--
Path: CHANGELOG.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Changelog

## 0.3.1 — Unreleased: Multimodal retrieval / RAG

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
