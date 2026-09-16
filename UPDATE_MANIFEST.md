<!--
Path: UPDATE_MANIFEST.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Update manifest

Base main commit: `a069d5b96eb0adb59b8c5208d1eaf68e4b191ed5`. Branch: `feat/qwen-multimodal-retrieval-v03`.
Package metadata: 0.2.1 → 0.3.0. 48 added/modified files; no deletions.

The ZIP is an overlay of these paths only. Apply it to a clean checkout of the
base commit on a feature branch, inspect the diff, and run the documented gates.
It is not a complete repository and includes no model weights or dependencies.
Existing teaching assets and unchanged code must come from the repository.
The report's real-model and runtime limitations remain release gates.

| Status | Repository path | Why changed |
|---|---|---|
| Added | `.dockerignore` | Exclude models, data, credentials, caches and build artifacts from Docker context. |
| Modified | `.env.example` | Update the documented compatibility release to v0.3.0. |
| Modified | `.github/workflows/ci.yml` | Run lightweight deterministic matrix; smoke-import new public classes; exclude real_model. |
| Added | `.github/workflows/runtime-compatibility.yml` | Manual Python 3.11–3.13 CPU optional-runtime import and dependency validation. |
| Modified | `.gitignore` | Ignore local credentials and editor swap files. |
| Modified | `CHANGELOG.md` | Describe v0.3 behavior and clarify inconsistent historical version heading. |
| Modified | `CONTRIBUTING.md` | Repair truncated contribution guide and document deterministic gates. |
| Modified | `Dockerfile` | Install multimodal retrieval and YAML extras without model weights. |
| Modified | `README.md` | Document implemented APIs, examples, migration, offline/config workflow and roadmap. |
| Added | `UPDATE_MANIFEST.md` | List every changed file and explain review archive application. |
| Added | `V03_REVIEW_REPORT.md` | Provide audit, scope decisions, validation evidence, limitations, roadmap and PR draft. |
| Added | `configs/qwen_multimodal_rag.example.yaml` | Provide a schema-v1 conservative runtime profile. |
| Modified | `docker-compose.yml` | Mount runtime configuration read-only alongside models/data/cache. |
| Modified | `docs/architecture.md` | Explain new validation, backend, multimodal orchestration and config boundaries. |
| Added | `docs/multimodal-retrieval.md` | Record official API evidence, resource limits and deployment constraints. |
| Modified | `docs/rag-patterns.md` | Describe implemented text/image pipeline and mark video as future work. |
| Modified | `docs/testing.md` | Document exact deterministic/opt-in commands and validation limits. |
| Modified | `examples/qwen_multimodal_retrieval.py` | Migrate example to explicit typed image objects. |
| Modified | `notebooks/03_multimodal_rag.ipynb` | Migrate one example image input to the strict contract. |
| Modified | `pyproject.toml` | Version 0.3.0, retrieval/config extras, Transformers v5 floor, test dependencies and marker. |
| Modified | `scenarios/scenario_06_true_multimodal_retrieval.py` | Migrate scenario image batches to explicit image objects. |
| Modified | `src/vlm_engineering/__init__.py` | Export QwenVLEmbedder, QwenVLReranker and MultimodalRAGPipeline. |
| Modified | `src/vlm_engineering/cli.py` | Extend existing argparse routing and classify expected user-facing failures. |
| Modified | `src/vlm_engineering/exceptions.py` | Introduce ValueError-compatible InputValidationError for precise CLI handling. |
| Added | `src/vlm_engineering/inputs.py` | Validate explicit text/image input, decode local images and normalize paths. |
| Added | `src/vlm_engineering/operational.py` | Map recognized dependency/resource/Hub failures; propagate unknown errors. |
| Modified | `src/vlm_engineering/qwen/downloader.py` | Map recognized download failures and use classified validation errors. |
| Modified | `src/vlm_engineering/qwen/model.py` | Compatible generation loader, multi-image ordering, local decode preflight and inference mode. |
| Modified | `src/vlm_engineering/qwen/registry.py` | Classify invalid user model selection without hiding backend ValueErrors. |
| Modified | `src/vlm_engineering/retrieval/__init__.py` | Export new names while preserving original aliases. |
| Added | `src/vlm_engineering/retrieval/backend.py` | Share lazy offline-aware constructor and reference cleanup logic. |
| Modified | `src/vlm_engineering/retrieval/in_memory.py` | Explicit clearing, finite/nonzero shape validation and stable cosine order. |
| Added | `src/vlm_engineering/retrieval/model_options.py` | Typed selectors, static dimensions, structural snapshot preflight. |
| Modified | `src/vlm_engineering/retrieval/multimodal_embedding.py` | Harden existing adapter with options, dimensions, batches and checked vectors. |
| Added | `src/vlm_engineering/retrieval/multimodal_rag.py` | Two-stage retrieval with atomic state replacement and all original evidence images. |
| Modified | `src/vlm_engineering/retrieval/rag.py` | Fix legacy empty re-index clearing and state publication after successful indexing. |
| Modified | `src/vlm_engineering/retrieval/reranker.py` | Harden existing scorer with raw/sigmoid semantics, output validation and stable ranks. |
| Added | `src/vlm_engineering/runtime_config.py` | Typed runtime dataclasses, safe YAML parsing, strict keys/types and lazy pipeline factory. |
| Modified | `tests/contract/test_public_api.py` | Include new CLI commands in the public contract. |
| Modified | `tests/contract/test_python_compatibility.py` | Assert v0.3.0 and updated all-extra dependency while retaining Python checks. |
| Added | `tests/contract/test_retrieval_release.py` | Assert aliases, extras, Transformers floor, coverage gate and deployment exclusions. |
| Added | `tests/e2e/test_qwen_real_models.py` | Add two opt-in real-model tests with local/offline environment variables. |
| Added | `tests/integration/test_cli_retrieval.py` | Test parser/routing, JSON results, preflight, downloads and error propagation. |
| Added | `tests/unit/test_multimodal_rag_v03.py` | Test ranking, provenance, query/evidence images, empty/repeated/failed indexing and vector validity. |
| Modified | `tests/unit/test_qwen_model_extended.py` | Use valid PNG fixtures; test bounded multi-image generation and unhidden errors. |
| Added | `tests/unit/test_qwen_retrieval_v03.py` | Deep adapter/model selection/output/resource failure and input-boundary tests. |
| Modified | `tests/unit/test_retrieval_extended.py` | Adapt injected backend expectations to documented options and explicit inputs. |
| Added | `tests/unit/test_runtime_config.py` | Strict YAML syntax/key/type/selection/bounds and model-free construction tests. |
