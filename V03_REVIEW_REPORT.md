<!--
Path: V03_REVIEW_REPORT.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# v0.3.0 implementation review

Review date: 2026-09-10. Branch: `feat/qwen-multimodal-retrieval-v03`.
Base: `a069d5b96eb0adb59b8c5208d1eaf68e4b191ed5` (GitHub main fetched and pulled).
Status: implementation and deterministic validation complete; **real-model/runtime
release certification remains outstanding**. No commit, push, release, PyPI
publication or merge was performed.

## 1. Audit of current main

Cloned the requested repository with main selected, then ran
`git pull --ff-only origin main` (already up to date). The starting package was
**0.2.1**, not the previously proposed v0.3 implementation. Read the source modules,
existing tests, scenarios/examples, required project files and workflows; inspected
notebook code and inventoried binary teaching/diagram assets. Existing main already
contained thin Qwen embedding and reranking adapters. Baseline on Python 3.12.14:
**116 passed in 2.99s; 86.26% branch-aware coverage**. The first attempt found pytest
missing; after installing development tooling, the baseline ran before source changes.

## 2. Current package architecture

| Layer | Audited main | Implemented change |
|---|---|---|
| Core | NumPy/Pillow, lazy ML imports, exceptions/constants | Shared input validation and classified operational failures |
| CLIP | Text/image encoding, similarity, zero-shot labels | Preserved |
| Qwen | Instruct registry 2B/4B/8B, loader, downloader | Early local image decoding, compatible auto class, bounded multi-image generation, inference mode |
| Documents | Dataclasses, native/visual fusion, chunk building | Existing VisualChunk schema retained |
| Retrieval | TextEmbedder, thin Qwen adapters, replacing cosine index | Validated Qwen adapters, stable ranking, explicit clear and vector checks |
| RAG | Text retrieval then first-image generation | Separate multimodal pipeline using all selected evidence; legacy empty-index fix |
| Interfaces | argparse CLI, Python examples/scenarios/notebooks | Extended existing CLI; optional typed runtime YAML |
| Operations | Docker/Compose, all-extras CI matrix | Excluded model/data context; lightweight CI plus manual CPU runtime matrix |

## 3. Problems discovered

- Empty legacy RAG re-indexing retained old vectors; re-index failures could desynchronize chunks/state.
- No adapter input validation, native dimension/count checks, finite/nonzero checks or score count checks.
- Bare image strings delegated modality detection to upstream, allowing ambiguous or invalid inputs.
- Retrieval wrappers lacked presets, local snapshot preflight, explicit offline/device/cache/revision controls and conservative batches.
- No end-to-end multimodal RAG orchestration; generator only accepted one image.
- Existing generation loading caught every exception as ModelLoadError, hiding programming bugs.
- Generation used an auto class absent from the declared Transformers 4.57 minimum.
- The ST reranker path needs Transformers v5+, beyond the direct-Qwen model-card minimum.
- CONTRIBUTING ended inside an unfinished shell block. Changelog's historical 1.0.0 heading conflicted with metadata 0.2.1.
- Docker lacked .dockerignore; .env was not ignored. A pre-existing tracked workflow swap file exists on main and is not part of this update archive.
- Existing test docs claimed Docker/notebook CI checks that the workflow did not perform.
- Separate, unmodified follow-up issue: TextChunker's paragraph-join/overlap branch can exceed max_chars after appending a new large paragraph. This warrants a focused follow-up fix; it is not used by the new multimodal pipeline.

## 4. Upstream research findings

Official sources were checked live; the Sentence Transformers **v5.4.0** source
was also cloned at `fe9361218c10b2ee18f497d73863788a6b592210` to verify minimum-version
API signatures and conditional imports.

- Qwen embedding supports text/image mixtures and Matryoshka output dimensions:
  64–2048 for 2B, up to 4096 for 8B. The wrappers preserve native-shape validation,
  truncate, then normalize. [2B model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B),
  [8B model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B).
- Qwen rerankers support CrossEncoder pair prediction with mixed modalities.
  Identity activation yields raw scores; sigmoid yields bounded scores without
  establishing calibration. [2B reranker](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B),
  [8B reranker](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B).
- ST v5.4 introduced the unified multimodal path; `[image]` is the documented
  modality extra. [Official integration guide](https://huggingface.co/blog/multimodal-sentence-transformers).
- The v5.4 `any-to-any` module conditionally imports AutoModelForMultimodalLM
  and raises a v5+ requirement when unavailable. This is why qwen-retrieval/all
  now require Transformers >=5.0. [Tagged upstream source](https://github.com/huggingface/sentence-transformers/blob/v5.4.0/sentence_transformers/base/modules/transformer.py).
- CrossEncoder supports cache_folder, revision, device, offline loading, prompt
  and activation_fn; SentenceTransformer exposes encode and dimension inspection.
  [CrossEncoder API](https://sbert.net/docs/package_reference/cross_encoder/model.html),
  [SentenceTransformer API](https://sbert.net/docs/package_reference/sentence_transformer/model.html).
- AutoModelForImageTextToText supports Qwen3-VL at the existing generator's 4.57
  floor. [Versioned Transformers reference](https://huggingface.co/docs/transformers/v4.57.0/en/model_doc/auto#transformers.AutoModelForImageTextToText).

No upstream implementation code was copied into this package. Upstream model
weights and their licenses remain separate.

## 5. Recommended v0.3.0 scope

Ship a coherent, validated multimodal retrieval/RAG layer, retaining the current
package boundaries and dependency-light imports. Treat the branch as reviewable
v0.3.0 work, pending real hardware validation before release. This decision was
explained before implementation, after the baseline and upstream audit.

## 6. Accepted candidate features

Validated 2B/8B embedding and reranking, custom/local selection, offline loading,
device/cache/revision, safe batch defaults, dimensions/normalization, stable ranks,
strict text/image contracts, resource/error guidance, runtime YAML, CLI extensions,
local retrieval-model downloads, original-evidence RAG, deep deterministic tests,
opt-in 2B real-model tests, compatible Python metadata, CI, Docker and docs.

## 7. Changed, rejected or postponed features

| Decision | Reason |
|---|---|
| Harden existing adapters and retain aliases | Main already had QwenMultimodalEmbedder/Reranker; avoid parallel implementations |
| Add bounded generate_images (1–16) now | Required to pass every selected RAG evidence image; single-image generate remains |
| Require explicit image objects in retrieval | Prevent bad image paths being silently treated as text |
| Local images only for retrieval | Deterministic preflight/offline behavior; remote fetching, video and backend-specific objects need separate contracts |
| Keep generator provided by application in YAML factory | Avoid duplicating the established Qwen generator configuration in a retrieval profile |
| Structural local snapshot checks only | Hashing/parsing every large tensor is expensive; backend diagnoses tensor corruption |
| No automatic Accelerate offloading policy | Device/RAM/disk choices are deployment-specific; configured backend injection remains available |
| Keep raw reranker scores by default | Preserves the old adapter behavior; normalized scoring is explicit |
| Keep core CI lightweight; move optional runtime imports to manual matrix | Avoid large framework downloads on ordinary PR jobs while retaining a runtime compatibility gate |
| Defer video, serving, databases, quantization and telemetry | Require independent lifecycle, integration and quality design; ranked below |

## 8. Implementation completed

Added shared input/model options/backend/error helpers; rewrote both retrieval
adapters; added MultimodalRAGPipeline and RuntimeConfig; extended argparse commands
and downloads; hardened index state and numeric inputs; added multi-image generator
support and inference mode; updated docs, deployment and tests. Old model imports
remain aliases, not copied code paths. State replacement is atomic for synchronous
use; concurrent mutation is not supported.

## 9–10. New and modified files

`UPDATE_MANIFEST.md` lists **every** added/modified path, its status and rationale.
The change-only ZIP contains those repository-relative paths, including this report
and the manifest. No deletions are required. Use the named base commit for review;
this is an overlay archive, not a full repository clone.

## 11. Dependency changes

- Package version 0.2.1 → 0.3.0. Python remains >=3.11,<3.14 with all three classifiers.
- New qwen-retrieval extra: sentence-transformers[image]>=5.4,<7,
  transformers>=5.0,<6, torch>=2.8,<3, torchvision>=0.23,<1,
  accelerate>=1.10,<2, huggingface-hub>=0.34,<2, qwen-vl-utils>=0.0.14,<1.
- New config extra: PyYAML>=6.0.2,<7.
- all now includes PyYAML, the ST image extra and Transformers >=5.0.
- dev now includes huggingface-hub (existing downloader tests import it), PyYAML
  and types-PyYAML. No torch/transformers are needed for deterministic CI.
- Existing core requirements and matching torch/torchvision lower bounds retained.
  No additional direct pins of transitive runtime dependencies.

## 12. Public API changes

```python
from vlm_engineering import QwenVLEmbedder, QwenVLReranker, MultimodalRAGPipeline, QwenVLModel
from vlm_engineering.runtime_config import load_runtime_config
```

QwenVLEmbedder: encode, embed_text, embed_image, unload.
QwenVLReranker: score, rerank, unload. Ranked results use the existing SearchResult.
MultimodalRAGPipeline: index_chunks, retrieve, answer.
QwenVLModel: new generate_images; existing generate preserved.
CLI: embed, rerank, validate-config; download-model adds embedding-size/reranker-size.
InputValidationError subclasses ValueError and the package base error, allowing
concise CLI reporting without catching arbitrary backend ValueErrors.

## 13. Backwards compatibility impact

Old adapter names/locations and positional model_id work. Plain text inputs work.
Bare image paths must migrate to explicit {"image": path}; remote retrieval images,
video and generic backend objects are rejected. Embeddings now normalize by default,
use batch size 1 and validate dimensions/finiteness/norm. Injected backends must accept
the documented upstream keyword arguments; custom embedders report native dimension.
Local generation now decodes images before inference, rejecting corrupt placeholders.
The legacy VisualRAGPipeline remains text-based and keeps its generation behavior,
with corrected re-index state. InMemoryVectorIndex now rejects invalid top_k and
malformed/nonfinite/zero vectors rather than coercing them.

## 14. Unit tests added

New retrieval adapter, multimodal RAG and runtime config test modules cover presets,
custom/local snapshots, offline constructor forwarding, invalid selectors/types,
dimension boundaries, native counts/shapes, normalization, text/image/mixed batches,
missing/corrupt/unsupported files, finite/nonzero vectors, raw/sigmoid scores,
stable ranking and original items, empty batches, operational errors, unknown-error
identity, re-index replacement/clear/failure atomicity, provenance, all images,
query images, token forwarding, config syntax/schema/type validation and lazy factories.
Existing generator tests now use actual tiny PNGs and include multi-image order,
batch limits, preflight and programming-error propagation. Total deterministic
case count grew from **116 to 340 (+224)**, including parametrized cases.

## 15. Contract and CLI tests added

Exports/aliases, version, Python classifiers/range, extras and minimum dependencies,
coverage floor, CI model exclusion, Docker model exclusion; CLI parser conflicts,
routing, original JSON candidate output, normalized scores, validation without
model imports, downloads and actionable errors. Final suite composition:
283 unit, 26 integration, 24 contract and 7 original top-level cases.

## 16. E2E tests added

Two opt-in real tests for Qwen3-VL-Embedding-2B and Qwen3-VL-Reranker-2B under
`tests/e2e/test_qwen_real_models.py`. They support explicit local paths, text/image/
mixed inputs, repeatability/normalization, raw/sigmoid consistency and cleanup.
The real_model marker and VLM_RUN_REAL_MODEL_TESTS=1 gate are both required for
execution. Ordinary collection has no model import/download side effects.

## 17–22. Exact quality gate results

Run in the isolated .venv with Python 3.12.14; commands below assume activation.

| Gate | Result |
|---|---|
| Baseline pytest | 116 passed in 2.99s |
| Final `python -m pytest -m "not real_model" --cov=vlm_engineering --cov-report=term-missing` (also emitted JUnit XML) | 340 passed, 2 deselected in 4.14s |
| Baseline → final branch-aware coverage | 86.26% → **94.82%**; original 80% threshold unchanged |
| `python -m ruff check .` | All checks passed |
| `python -m mypy .` | Success: no issues found in 68 source files |
| `python -m pip check` | No broken requirements found |
| `python -m pip_audit .` | No known vulnerabilities found (core project dependency resolution) |
| E2E gate without opt-in | 2 skipped in 0.12s; no real model execution |
| `git diff --check` | Passed |
| Wheel build | 0.3.0 wheel built; 35 code/metadata entries, no weights/images/notebooks |

The exact final coverage output is appended below. The command was executed with
`.venv/bin/python` in this environment. No coverage exclusions or thresholds were
relaxed to pass. The core audit does not certify the optional ML dependency stack.

## 23. Not tested and why

- No real Qwen weights were loaded: this environment was used for deterministic
  development, not a provisioned model/GPU evaluation run. There is no evidence here
  of quality, performance or peak RAM/VRAM on real hardware.
- No real multi-image generator E2E; only its processor/generation contract is tested.
- Python 3.11 and 3.13 were not available in the executed environment. Their metadata
  and CI/manual runtime matrices are retained; do not claim local execution on them.
- Heavy optional ML imports/full extras installation were not executed locally.
  Tagged source/API inspection and fake factories are not substitutes for that gate.
- Docker/Compose were not available (`command -v docker` returned no executable).
  No container build, GPU passthrough or mount-permission test was run.
- OS OOM-killer termination cannot be caught as a Python error. Offloading and
  alternative custom snapshot layouts need deployment-specific validation.
- No PyPI, GitHub release or deployment workflows were triggered.

## 24. Exact real-model commands

```bash
python -m pip install -e ".[qwen-retrieval,dev]"
vlm-lab download-model --embedding-size 2b --output models/Qwen3-VL-Embedding-2B
vlm-lab download-model --reranker-size 2b --output models/Qwen3-VL-Reranker-2B
HF_HUB_OFFLINE=1 \
VLM_RUN_REAL_MODEL_TESTS=1 \
VLM_QWEN_EMBEDDING_MODEL_PATH=models/Qwen3-VL-Embedding-2B \
VLM_QWEN_RERANKER_MODEL_PATH=models/Qwen3-VL-Reranker-2B \
pytest -m real_model tests/e2e -v
```

Hub-backed alternative (unset local-path variables and HF_HUB_OFFLINE):

```bash
VLM_RUN_REAL_MODEL_TESTS=1 pytest -m real_model tests/e2e -v
```

## 25. Resource requirements and caveats

Upstream guidance is approximately 8 GB VRAM for 2B and 20 GB for 8B per model,
workload-dependent. Three-model RAG can exceed those figures substantially.
CPU execution can be extremely slow. [Official ST guide](https://huggingface.co/blog/multimodal-sentence-transformers).

Weight-only arithmetic at BF16/FP16 is approximately 4 GB for 2B and 16 GB for 8B
per model, before activations, framework allocations and image/context work. Budget
additional disk for snapshots/temp files/cache duplication. Batch size 1 does not
make a model fit inadequate hardware. Multi-image input is bounded at 16 attachments,
including a query image, but that is a contract limit, not a memory guarantee.

unload drops wrapper references only; external references/allocator caches may
retain memory. Advanced backend injection can configure dtype/device_map/offload
folders. CPU/disk offloading adds I/O and host memory requirements.
[Accelerate guide](https://huggingface.co/docs/accelerate/usage_guides/big_modeling).
Models are mounted/downloaded separately; no global cache deletion occurs.

## 26. Recommended v0.4.0 roadmap

The ratings below are engineering judgments for this repository, informed by the
current source and linked upstream docs. H/M/L mean high/medium/low. For effort and
maintenance, lower is better; for value/fit/maturity/testability, higher is better.
Hardware describes incremental requirements beyond model inference.

| Rank | Feature | Value | Fit | Effort | Maintenance | Upstream maturity | Testability | Hardware |
|---|---|---|---|---|---|---|---|---|
| 1 | Retrieval/grounding benchmark fixtures, Recall@K, nDCG, citation checks | H | H | M | M | H | H | CPU metrics; GPU runs |
| 2 | Persistent index protocol plus one Qdrant adapter | H | H | M | M | H | H | CPU/disk/service |
| 3 | Model/input/revision fingerprints and cache invalidation | H | H | M | M | H | H | CPU/disk |
| 4 | Explicit resource profiles, image-token budgets, staged unload | H | H | M | M | H | M | GPU validation |
| 5 | Stage latency/memory hooks and optional OpenTelemetry | M | H | L–M | L–M | H for traces/metrics | H | CPU |
| 6 | Native PDF/OCR parser interfaces and chunk-size bug fix | H | H | H | H | H, integration varies | H with fixtures | CPU; OCR varies |
| 7 | Quantized inference presets | H | M | H | H | H upstream, model-specific | M | Hardware matrix |
| 8 | Bounded batch inference and cancellable async facade | M | H | M–H | M | H | M–H | GPU capacity |
| 9 | Streaming generation/model-serving API | M | M | H | H | H upstream | M | GPU/service |
| 10 | FAISS/pgvector additional adapters | M | H | M each | M each | Established | H | CPU/database |
| 11 | Video understanding and temporal retrieval | M | M | H | H | Model support exists | M–L | High GPU/storage |
| 12 | Grounding boxes, spatial evaluation, GUI/visual-agent perception and visual coding | Domain-specific H | M | H | H | Model capability varies | L–M | GPU + labeled fixtures |

**Recommended v0.4 scope:** benchmark/evaluation contracts, persistent-index
protocol with one Qdrant adapter, provenance-aware cache keys, basic stage metrics
and bounded resource profiles. Fix the independently found chunk-size issue in a
small separate change. Defer additional databases until the protocol is proven;
defer video/agents/serving until evaluation and resource budgets are reliable.

Evidence: Qdrant's official quickstart supports collection/vector/payload-backed
retrieval and persisted local storage, fitting the existing SearchResult/VisualChunk
boundary. [Qdrant](https://qdrant.tech/documentation/quickstart/).
Transformers documents 4/8-bit bitsandbytes integration with hardware conditions,
which makes quantization promising but demands hardware-specific tests.
[Quantization](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes).
OpenTelemetry Python offers established trace/metric APIs for optional stage
instrumentation. [OpenTelemetry](https://opentelemetry.io/docs/languages/python/).
Qwen3-VL has upstream vision/video capabilities, but package input contracts and
quality fixtures must come first. [Qwen3-VL](https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl).

## 27. Recommended commit message

```text
feat(retrieval): add validated Qwen multimodal RAG for v0.3.0
```

## 28. Recommended PR title

```text
Add validated Qwen multimodal retrieval and evidence-preserving RAG
```

## 29. Recommended PR description

Main's Qwen retrieval wrappers delegated unchecked inputs/outputs to upstream,
and its RAG path only embedded text and supplied one image to generation. Empty
re-indexing could retain stale evidence. These gaps made multimodal workflows
unreliable and local/offline deployment difficult to diagnose.

This PR hardens the existing adapters, adds 2B/8B/custom/local loading controls,
strict local text/image contracts, dimensions and score validation, and introduces
a multimodal pipeline with atomic re-indexing, stable reranking and all selected
original images. It adds typed YAML profiles, CLI commands, retrieval downloads,
classified operational errors, multi-image generation and deployment guidance.
Existing adapter names remain aliases; bare image paths now require explicit
image objects and embeddings normalize by default.

The ST reranker requires Transformers v5+, so qwen-retrieval/all use that floor.
Normal CI remains lightweight across Python 3.11–3.13; manual runtime compatibility
and explicitly gated real-model tests handle heavier checks.

Validation: 340 deterministic tests passed; 94.82% coverage with the unchanged
80% gate; Ruff, mypy, pip check and core pip-audit passed. Wheel contents exclude
weights. Real model/GPU inference, optional-runtime imports, Docker and local
Python 3.11/3.13 execution remain outstanding. No release or publication action
is included.

## 30–31. Change archive and manifest

The ZIP includes only added/modified files against the named main commit.
`UPDATE_MANIFEST.md` is included and lists each file's purpose. The repository
working tree remains on the feature branch with reviewable uncommitted changes.
No weights, caches, environment directories, build products, secrets or tokens
were added to the review archive.

## Appendix: final deterministic test output

```text
........................................................................ [ 21%]
........................................................................ [ 42%]
........................................................................ [ 63%]
........................................................................ [ 84%]
....................................................                     [100%]
================================ tests coverage ================================
_______________ coverage: platform linux, python 3.12.14-final-0 _______________

Name                                              Stmts   Miss Branch BrPart  Cover   Missing
---------------------------------------------------------------------------------------------
src/vlm_engineering/__main__.py                       2      2      0      0     0%   9-11
src/vlm_engineering/cli.py                          156      4     12      2    95%   43-49, 193
src/vlm_engineering/clip/encoder.py                  63     16      8      3    70%   53-76, 90, 102
src/vlm_engineering/config.py                        11      2      0      0    82%   23-24
src/vlm_engineering/documents/chunking.py            41      4     16      2    89%   22, 40-42
src/vlm_engineering/documents/fusion.py              20      0     14      1    97%   21->23
src/vlm_engineering/qwen/downloader.py               22      2      6      0    93%   51-52
src/vlm_engineering/qwen/model.py                   110     11     46      3    90%   58, 61, 72, 75-78, 180-181, 196-206
src/vlm_engineering/qwen/registry.py                 34      1      8      1    95%   79
src/vlm_engineering/retrieval/multimodal_rag.py      81      0     20      1    99%   39->41
src/vlm_engineering/retrieval/text_embedding.py      24      8      4      1    61%   28-38
src/vlm_engineering/utils.py                         25      3     10      3    83%   20, 23, 39
---------------------------------------------------------------------------------------------
TOTAL                                              1176     53    330     17    95%

17 files skipped due to complete coverage.
Required test coverage of 80.0% reached. Total coverage: 94.82%
340 passed, 2 deselected in 4.14s
```
