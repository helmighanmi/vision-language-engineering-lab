<!--
Path: README.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Vision-Language Engineering Lab

**Production-oriented Vision-Language engineering with CLIP, configurable Qwen3-VL, document understanding, multimodal retrieval, reranking, and RAG.**

`vision-language-engineering-lab` is a Python package for building and studying Vision-Language applications with reusable engineering patterns rather than notebook-only demos.

The implementation lives under `src/vlm_engineering/`. Notebooks are analysis/demo clients, `examples/` provides small API examples, and `scenarios/` contains end-to-end application recipes.

> **v0.3.1 compatibility target:** Python 3.11, 3.12, and 3.13.

v0.3.1 expands the project from generative VLM and text-oriented visual RAG into a **two-stage multimodal retrieval stack**:

```text
query
  ↓
Qwen3-VL-Embedding
  ↓
vector retrieval / candidate_k
  ↓
Qwen3-VL-Reranker
  ↓
top_k multimodal evidence
  ↓
original text + images
  ↓
Qwen3-VL generation
  ↓
grounded answer
```

Contributions are welcome. See [Contributing](#20-contributing) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

---
### Interactive architecture map

Explore the package structure, services, and multimodal RAG flow:

[Open the interactive architecture map](https://helmighanmi.github.io/vision-language-engineering-lab/)
---
## 1. Quick start

### Install from PyPI

Core package:

```bash
python -m pip install vision-language-engineering-lab
```

Qwen3-VL generation:

```bash
python -m pip install "vision-language-engineering-lab[qwen]"
```

Qwen multimodal embedding and reranking:

```bash
python -m pip install "vision-language-engineering-lab[qwen-retrieval]"
```

All runtime features:

```bash
python -m pip install "vision-language-engineering-lab[all]"
```

### Install from the repository

```bash
git clone https://github.com/helmighanmi/vision-language-engineering-lab.git
cd vision-language-engineering-lab

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[all]"
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[all]"
```

### Example 1 — Real-image smoke test with Qwen3-VL 2B

A repository clone includes `data/diagram_random_clean.png` as a real-image smoke-test asset:

```bash
vlm-lab describe data/diagram_random_clean.png \
  --model-size 2b \
  --prompt "Describe this image accurately. List the main objects, text, and important visual relationships."
```

Equivalent Python API:

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel(model_size="2b")

answer = model.generate(
    "data/diagram_random_clean.png",
    (
        "Describe this image accurately. "
        "List the main objects, text, and important visual relationships."
    ),
)

print(answer)
```

The default generative model is:

```text
Qwen/Qwen3-VL-2B-Instruct
```

If you installed from PyPI rather than cloning the repository, use your own image path:

```bash
vlm-lab describe /path/to/your/image.png \
  --model-size 2b \
  --prompt "Describe this image accurately."
```

### Example 2 — Multimodal embeddings

```python
from vlm_engineering import QwenVLEmbedder

embedder = QwenVLEmbedder(
    model_size="2b",
    dimensions=1024,
)

text_vector = embedder.embed_text(
    "Find an architecture diagram containing a cache."
)

image_vector = embedder.embed_image(
    "data/diagram_random_clean.png"
)

mixed_vectors = embedder.encode(
    [
        {"text": "Redis caching architecture"},
        {"image": "data/diagram_random_clean.png"},
        {
            "text": "Architecture diagram",
            "image": "data/diagram_random_clean.png",
        },
    ]
)

print(text_vector.shape)
print(image_vector.shape)
print(mixed_vectors.shape)
```

The retrieval input contract is intentionally explicit:

```python
{"text": "..."}
{"image": "..."}
{"text": "...", "image": "..."}
```

Unknown fields, empty inputs, missing local images, corrupt files, and unsupported inputs are rejected before expensive model inference whenever possible.

### Example 3 — Multimodal reranking

```python
from vlm_engineering import QwenVLReranker

documents = [
    {"text": "Redis provides application caching."},
    {"text": "PostgreSQL stores persistent application data."},
    {"image": "data/diagram_random_clean.png"},
]

reranker = QwenVLReranker(model_size="2b")

scores = reranker.score(
    "Find the evidence related to caching.",
    documents,
)

ranked = reranker.rerank(
    "Find the evidence related to caching.",
    documents,
    top_k=2,
)

print(scores)
for result in ranked:
    print(result.rank, result.score, result.item)
```

Raw scores and normalized scores are both supported. Normalized reranker scores are relevance transformations, not calibrated probabilities.

### Example 4 — True multimodal RAG

```python
from vlm_engineering import (
    MultimodalRAGPipeline,
    NativePageContent,
    QwenVLEmbedder,
    QwenVLModel,
    QwenVLReranker,
    VisualAnalysis,
    VisualChunkBuilder,
)

chunk = VisualChunkBuilder().build(
    NativePageContent(
        document_id="demo",
        page=1,
        source_file="architecture.pdf",
        image_ref="data/diagram_random_clean.png",
    ),
    VisualAnalysis(
        page_type="diagram",
        title="Architecture",
        summary="Technical architecture diagram.",
    ),
)

pipeline = MultimodalRAGPipeline(
    embedder=QwenVLEmbedder(model_size="2b"),
    generator=QwenVLModel(model_size="2b"),
    reranker=QwenVLReranker(model_size="2b"),
    candidate_k=12,
    top_k=3,
    max_new_tokens=256,
)

pipeline.index_chunks([chunk])

answer = pipeline.answer(
    "Which components are connected?"
)

print(answer.answer)
```

The pipeline performs:

```text
multimodal embedding
    ↓
candidate retrieval
    ↓
multimodal reranking
    ↓
selected original evidence
    ↓
Qwen3-VL generation
```

### Example 5 — Select 2B / 4B / 8B generative Qwen3-VL

```python
from vlm_engineering import QwenVLModel

model_2b = QwenVLModel(model_size="2b")
model_4b = QwenVLModel(model_size="4b")
model_8b = QwenVLModel(model_size="8b")

print(model_2b.model_source)
print(model_4b.model_source)
print(model_8b.model_source)
```

| Preset | Hugging Face model | Typical use |
|---|---|---|
| `2b` | `Qwen/Qwen3-VL-2B-Instruct` | default development / lower-resource inference |
| `4b` | `Qwen/Qwen3-VL-4B-Instruct` | balanced quality/resource option |
| `8b` | `Qwen/Qwen3-VL-8B-Instruct` | stronger hardware / higher-capacity inference |

### Example 6 — Local/offline model loading

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel.from_local(
    "models/Qwen3-VL-2B-Instruct"
)

answer = model.generate(
    "data/diagram_random_clean.png",
    "Describe the diagram using only the visible evidence.",
)

print(answer)
```

`from_local()` uses local-files-only loading and does not silently fall back to the Hugging Face Hub.

### Example 7 — Structured document analysis

```python
from vlm_engineering import QwenVLModel
from vlm_engineering.documents import analyze_visual_document

model = QwenVLModel(model_size="4b")

analysis = analyze_visual_document(
    model,
    "data/architecture.png",
)

print(analysis.to_dict())
```

The structured result can contain page type, title, summary, entities, relations, important text, and uncertainties.

### Example 8 — Text-only visual RAG

When an existing RAG platform supports only text embeddings, the VLM can first turn visual evidence into faithful retrieval text:

```python
from vlm_engineering.retrieval import TextEmbedder

embedder = TextEmbedder()

embeddings = embedder.encode(
    [
        "Redis provides application caching.",
        "PostgreSQL stores persistent application data.",
    ]
)

print(embeddings.shape)
```

### Example 9 — CLI

```bash
# List generative Qwen presets
vlm-lab models

# Describe an image
vlm-lab describe data/diagram_random_clean.png --model-size 2b

# Structured analysis
vlm-lab analyze data/architecture.png --model-size 4b

# Multimodal embedding
vlm-lab embed \
  --text "Find a cache architecture" \
  --model-size 2b \
  --dimensions 1024

# Reranking
vlm-lab rerank \
  --query "Find the cache architecture" \
  --documents-json candidates.json \
  --model-size 2b

# Validate a runtime profile without loading large models
vlm-lab validate-config \
  configs/qwen_multimodal_rag.example.yaml
```

For complete workflows, see [`scenarios/README.md`](scenarios/README.md).

---

## 2. What this project demonstrates

```text
CLIP foundations
   -> image/text embeddings and zero-shot similarity

Qwen3-VL Instruct
   -> image description
   -> VQA
   -> diagram/table/screenshot understanding
   -> structured visual analysis

Document understanding
   -> native/OCR evidence + visual semantics
   -> traceable VisualChunk objects

Text-only visual RAG
   -> VLM descriptions
   -> normal text embeddings
   -> retrieval
   -> original-image grounding

Multimodal retrieval
   -> Qwen3-VL-Embedding
   -> vector recall
   -> Qwen3-VL-Reranker
   -> top-k multimodal evidence

Multimodal RAG
   -> embedding
   -> candidate retrieval
   -> reranking
   -> original text/images
   -> Qwen3-VL answer
```

The project intentionally separates:

- reusable package code under `src/vlm_engineering/`
- deterministic tests under `tests/`
- small API examples under `examples/`
- end-to-end recipes under `scenarios/`
- exploratory notebooks under `notebooks/`
- architecture and operational documentation under `docs/`

---

## 3. Python compatibility

v0.3.1 targets:

| Python | Support |
|---|---:|
| 3.11 | ✅ Supported |
| 3.12 | ✅ Supported |
| 3.13 | ✅ Supported |

The package metadata declares:

```text
>=3.11,<3.14
```

GitHub Actions validates supported Python versions independently.

Ruff targets Python 3.11 syntax so package code does not accidentally depend on Python 3.12+ syntax.

### Google Colab

When the active Colab runtime Python version is supported:

```python
!pip install "vision-language-engineering-lab[all]"
```

Then:

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel(model_size="2b")
print(model.model_source)
```

For Qwen workloads, enable a GPU runtime where available.

---

## 4. Installation options

### Requirements

- Python **3.11, 3.12, or 3.13**
- Git for source installs
- enough disk space for selected models
- a GPU is strongly recommended for practical Qwen3-VL inference
- CPU-only execution may work for some paths but can be slow or require offloading

### Create a virtual environment

Linux/macOS/GitHub Codespaces:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### Choose what to install

Development/testing:

```bash
python -m pip install -e ".[dev]"
```

CLIP:

```bash
python -m pip install -e ".[clip]"
```

Qwen3-VL generation:

```bash
python -m pip install -e ".[qwen]"
```

Text retrieval:

```bash
python -m pip install -e ".[retrieval]"
```

Qwen multimodal retrieval:

```bash
python -m pip install -e ".[qwen-retrieval]"
```

Runtime YAML profiles:

```bash
python -m pip install -e ".[config]"
```

All runtime features:

```bash
python -m pip install -e ".[all]"
```

Full contributor environment:

```bash
python -m pip install -e ".[all,dev,notebooks]"
```

A compatibility `requirements.txt` is also provided:

```bash
python -m pip install -r requirements.txt
```

`pyproject.toml` remains the dependency source of truth.

Confirm the CLI:

```bash
vlm-lab --help
vlm-lab models
```

---

## 5. Qwen models used by this project

### Generative VLM

| Preset | Hugging Face model | Role |
|---|---|---|
| `2b` | `Qwen/Qwen3-VL-2B-Instruct` | default generative VLM |
| `4b` | `Qwen/Qwen3-VL-4B-Instruct` | balanced generative option |
| `8b` | `Qwen/Qwen3-VL-8B-Instruct` | higher-capacity generative option |

### Multimodal embedding

| Preset | Hugging Face model | Role |
|---|---|---|
| `2b` | `Qwen/Qwen3-VL-Embedding-2B` | multimodal recall / shared embedding space |
| `8b` | `Qwen/Qwen3-VL-Embedding-8B` | higher-capacity multimodal embeddings |

### Multimodal reranking

| Preset | Hugging Face model | Role |
|---|---|---|
| `2b` | `Qwen/Qwen3-VL-Reranker-2B` | second-stage candidate reranking |
| `8b` | `Qwen/Qwen3-VL-Reranker-8B` | higher-capacity reranking |

The size labels are parameter classes, not exact runtime-memory requirements.

Real inference also needs memory for image processing, activations, framework overhead, KV cache/generation state, and model-specific runtime structures.

`trust_remote_code` remains disabled by default where supported by the package.

---

## 6. Model selection and loading

### Generative model

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel(model_size="2b")
```

Explicit Hub model:

```python
model = QwenVLModel.from_hub(
    "Qwen/Qwen3-VL-4B-Instruct"
)
```

Explicit local model:

```python
model = QwenVLModel.from_local(
    "models/Qwen3-VL-4B-Instruct"
)
```

### Embedding model

```python
from vlm_engineering import QwenVLEmbedder

embedder = QwenVLEmbedder(
    model_size="2b",
    dimensions=1024,
)
```

### Reranker

```python
from vlm_engineering import QwenVLReranker

reranker = QwenVLReranker(
    model_size="2b",
)
```

Model selectors are intentionally explicit. Depending on the API/CLI path, use one of:

```text
model_size
model_id
model_path
```

Mutually exclusive model-source options prevent accidental fallback to a different model than the one requested.

---

## 7. Hugging Face cache and project-local models

Two storage modes are supported.

### Hub/cache mode

For quick experimentation:

```bash
vlm-lab describe \
  data/diagram_random_clean.png \
  --model-size 2b
```

The Hugging Face Hub cache is typically reused automatically on subsequent runs.

### Project-local generative model

```bash
vlm-lab download-model \
  --model-size 2b \
  --output models/Qwen3-VL-2B-Instruct
```

Then:

```bash
HF_HUB_OFFLINE=1 vlm-lab describe \
  data/diagram_random_clean.png \
  --model-path models/Qwen3-VL-2B-Instruct \
  --prompt "Describe this image accurately."
```

### Project-local embedding model

```bash
vlm-lab download-model \
  --embedding-size 2b \
  --output models/Qwen3-VL-Embedding-2B
```

Example offline use:

```bash
HF_HUB_OFFLINE=1 vlm-lab embed \
  --text "architecture diagram" \
  --model-path models/Qwen3-VL-Embedding-2B
```

### Project-local reranker

```bash
vlm-lab download-model \
  --reranker-size 2b \
  --output models/Qwen3-VL-Reranker-2B
```

The package does **not** delete a user's shared Hugging Face cache automatically.

Model directories under `models/` are intended for local deployment and remain outside source control.

---

## 8. Advanced model-loading options

### `trust_remote_code`

Default: **False**.

Only enable repository-provided Python code for model repositories you have reviewed and trust.

### Device / cache / revision

The Qwen wrappers support configurable model-loading behavior appropriate to their backend, including device selection, cache location, revision selection, and local-only loading.

For generative inference:

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel(
    model_size="4b",
    device_map="auto",
    dtype="auto",
)
```

For large multimodal retrieval models, conservative defaults are preferred. Embedding/reranking default to small batch sizes to reduce accidental memory pressure.

---

## 9. Core Qwen3-VL use cases

### Image captioning / rich description

```bash
vlm-lab describe photo.jpg \
  --prompt "Write a factual, detailed description."
```

### Visual question answering

```bash
vlm-lab describe dashboard.png \
  --model-size 4b \
  --prompt "Which metric increased the most and what visual evidence supports it?"
```

### Diagram / architecture understanding

```bash
vlm-lab analyze architecture.png \
  --model-size 4b
```

### Screenshot / UI understanding

Use `describe` or `analyze` for dashboards, application screenshots, forms, and technical UI states.

### Tables and document pages

Render a page as an image and request structured extraction. For production document workflows, preserve native/OCR text separately and use the VLM for layout, grouping, relationships, and visual semantics.

---

## 10. VLM + OCR/native parsing: hybrid by default

```text
Native extraction / OCR                  Qwen3-VL
exact text, IDs, numbers                 layout, arrows, groups,
long deterministic text                 relationships, visual meaning
             \                           /
              \                         /
                       fusion
                         |
                   RAG-ready chunk
```

A VLM can replace OCR in some workflows, but robust document pipelines often preserve native/OCR evidence for literal accuracy while adding VLM semantics where visual structure matters.

---

## 11. Build a RAG-ready visual chunk

```bash
vlm-lab chunk data/page_001.png \
  --model-size 4b \
  --document-id architecture-v1 \
  --source-file architecture.pdf \
  --page 1 \
  --native-text-file data/page_001.txt
```

The resulting `VisualChunk` preserves traceability such as document ID, source file, page, image reference, entities, relations, metadata, and fused retrieval text.

---

## 12. Visual RAG with a text embedder

```text
image/page
   ↓
Qwen3-VL description / structured analysis
   ↓
retrieval text
   ↓
Sentence Transformers text embedder
   ↓
vector index / vector database
```

This path is useful when an existing RAG platform supports only text embeddings.

```bash
python examples/text_only_visual_rag.py
```

The original `image_ref` is retained so the final VLM can inspect the original image rather than relying only on generated descriptions.

---

## 13. Multimodal retrieval and RAG — v0.3.1

### 13.1 Installation

```bash
python -m pip install -e ".[qwen-retrieval,config]"
```

The retrieval stack is designed around:

```text
Qwen3-VL-Embedding
        ↓
candidate recall
        ↓
Qwen3-VL-Reranker
        ↓
precision-oriented reranking
        ↓
Qwen3-VL generation
```

### 13.2 Multimodal input contract

The public retrieval contract accepts:

```python
{"text": "..."}
```

```python
{"image": "path/to/image.png"}
```

```python
{
    "text": "Find diagrams similar to this",
    "image": "path/to/query.png",
}
```

Plain text strings may be accepted as shorthand where documented.

Local images are validated before expensive inference where practical. Unknown fields, empty inputs, missing files, directories passed as images, corrupt files, and unsupported inputs fail early with readable errors.

### 13.3 Embeddings

```python
from vlm_engineering import QwenVLEmbedder

embedder = QwenVLEmbedder(
    model_size="2b",
    dimensions=1024,
)

text_vector = embedder.embed_text(
    "architecture diagram"
)

image_vector = embedder.embed_image(
    "data/diagram_random_clean.png"
)

vectors = embedder.encode(
    [
        {"text": "Redis caching"},
        {"image": "data/diagram_random_clean.png"},
        {
            "text": "Architecture",
            "image": "data/diagram_random_clean.png",
        },
    ]
)
```

Embeddings are normalized by default where configured by the wrapper.

Supported Qwen embedding presets expose the dimensions supported by their model contract. Invalid dimensions are rejected before inference.

### 13.4 Reranking

```python
from vlm_engineering import QwenVLReranker

documents = [
    {"text": "Redis provides caching."},
    {"text": "PostgreSQL stores persistent state."},
    {"image": "data/diagram_random_clean.png"},
]

reranker = QwenVLReranker(model_size="2b")

scores = reranker.score(
    "Find caching evidence",
    documents,
)

ranked = reranker.rerank(
    "Find caching evidence",
    documents,
    top_k=2,
)
```

Ranking preserves the original candidate object in each result.

Where normalized scoring is requested, the transformed score should be interpreted as a relevance score rather than a calibrated probability.

### 13.5 Full multimodal RAG

```python
from vlm_engineering import (
    MultimodalRAGPipeline,
    QwenVLEmbedder,
    QwenVLModel,
    QwenVLReranker,
)

pipeline = MultimodalRAGPipeline(
    embedder=QwenVLEmbedder(model_size="2b"),
    reranker=QwenVLReranker(model_size="2b"),
    generator=QwenVLModel(model_size="2b"),
    candidate_k=12,
    top_k=3,
    max_new_tokens=256,
)

pipeline.index_chunks(chunks)

answer = pipeline.answer(
    "How does the cache interact with the rest of the architecture?"
)

print(answer.answer)
```

Important pipeline behavior:

- `candidate_k >= top_k > 0`
- re-indexing replaces previous in-memory state
- empty re-indexing clears the previous collection
- failed indexing does not intentionally leave a half-built index
- original document/page/source metadata is preserved
- original image references remain available for final grounding
- retrieved evidence is reranked before generation
- missing visual evidence is reported explicitly when the generation path requires it

### 13.6 Multimodal retrieval queries

A query can include text and image evidence where supported:

```python
results = pipeline.retrieve(
    {
        "text": "Find architecture similar to this diagram.",
        "image": "data/query_diagram.png",
    }
)
```

### 13.7 CLI

Embedding text:

```bash
vlm-lab embed \
  --text "Find a cache diagram" \
  --model-size 2b \
  --dimensions 1024
```

Embedding an image:

```bash
vlm-lab embed \
  --image data/diagram_random_clean.png \
  --model-size 2b
```

Reranking:

```bash
vlm-lab rerank \
  --query "Find the architecture" \
  --documents-json candidates.json \
  --model-size 2b
```

### 13.8 Runtime YAML

Example profile:

```yaml
version: 1

embedding:
  model:
    model_size: 2b
    revision: null
    device: null
    cache_folder: null
    trust_remote_code: false

  batch_size: 1
  dimensions: 1024
  normalize_embeddings: true

reranker:
  model:
    model_size: 2b
    revision: null
    device: null
    cache_folder: null
    trust_remote_code: false

  batch_size: 1
  normalize_scores: true

rag:
  top_k: 3
  candidate_k: 12
  max_new_tokens: 256
```

Validate without loading model weights:

```bash
vlm-lab validate-config \
  configs/qwen_multimodal_rag.example.yaml
```

The configuration layer:

- uses safe YAML loading
- has an explicit schema version
- validates field types
- rejects unknown keys
- rejects conflicting model selectors
- validates embedding dimensions where possible without model loading
- rejects invalid `candidate_k` / `top_k` combinations

Example Python usage:

```python
from vlm_engineering import QwenVLModel
from vlm_engineering.runtime_config import load_runtime_config

config = load_runtime_config(
    "configs/qwen_multimodal_rag.example.yaml"
)

pipeline = config.build_pipeline(
    QwenVLModel(model_size="2b")
)
```

### 13.9 Reliability and operational errors

v0.3.1 strengthens user-facing validation around common operational failures, including:

- missing optional dependencies
- invalid multimodal input
- missing/corrupt image files
- invalid local model directories
- incomplete local model snapshots where detectable
- Hugging Face authentication/offline failures
- resource / out-of-memory failures
- disk-space failures
- invalid embedding dimensions
- unexpected embedding shapes
- NaN / Inf embeddings
- invalid reranker outputs
- NaN / Inf reranker scores
- malformed YAML profiles

Expected operational problems are mapped to readable errors where possible. Unexpected programming errors should remain visible for debugging rather than being silently swallowed.

### 13.10 Resource safety

Qwen multimodal retrieval models are large.

The retrieval wrappers use conservative batch defaults such as:

```text
batch_size = 1
```

Users with stronger hardware can tune batching after validating their memory envelope.

For practical workloads:

- GPU execution is recommended
- CPU execution can be substantially slower
- backend offloading may use CPU or disk depending on the model/backend
- multiple images and high-resolution inputs increase memory requirements
- local model storage requires several gigabytes depending on the selected model

### 13.11 Real-model validation

Large real-model tests are opt-in and excluded from normal PR CI.

Run:

```bash
VLM_RUN_REAL_MODEL_TESTS=1 \
python -m pytest \
  -m real_model \
  tests/e2e \
  -v
```

Using explicit local/offline models:

```bash
export VLM_QWEN_EMBEDDING_MODEL_PATH=\
models/Qwen3-VL-Embedding-2B

export VLM_QWEN_RERANKER_MODEL_PATH=\
models/Qwen3-VL-Reranker-2B

export VLM_RUN_REAL_MODEL_TESTS=1

python -m pytest \
  -m real_model \
  tests/e2e \
  -v
```

Normal CI relies on deterministic fake/injected backends rather than downloading multi-gigabyte weights.

### 13.12 Migration from v0.2.x

Where retained by the package, earlier retrieval class names remain compatibility aliases.

New code should prefer:

```python
from vlm_engineering import (
    QwenVLEmbedder,
    QwenVLReranker,
    MultimodalRAGPipeline,
)
```

For image inputs, prefer explicit multimodal objects:

```python
{"image": "path/to/image.png"}
```

rather than ambiguous bare path strings.

---

## 14. CLIP foundations

```python
from PIL import Image

from vlm_engineering import CLIPEncoder

image = Image.open(
    "data/diagram_random_clean.png"
).convert("RGB")

clip = CLIPEncoder()

result = clip.zero_shot_classify(
    image,
    ["cat", "dog", "airplane"],
)

print(result)
```

CLIP is useful for learning the shared image/text embedding-space idea that modern multimodal retrieval builds upon.

---

## 15. Examples and application scenarios

Small API examples:

```text
examples/clip_zero_shot.py
examples/qwen_describe_image.py
examples/qwen_model_selection.py
examples/qwen_local_model.py
examples/structured_visual_chunk.py
examples/text_only_visual_rag.py
examples/qwen_multimodal_retrieval.py
```

End-to-end scenarios:

```text
scenarios/scenario_01_image_captioning.py
scenarios/scenario_02_visual_question_answering.py
scenarios/scenario_03_diagram_to_json.py
scenarios/scenario_04_document_page_to_rag_chunk.py
scenarios/scenario_05_text_only_visual_rag.py
scenarios/scenario_06_true_multimodal_retrieval.py
scenarios/scenario_07_compare_qwen_presets.py
scenarios/scenario_08_hub_vs_local_loading.py
scenarios/scenario_09_batch_structured_analysis.py
```

See [`scenarios/README.md`](scenarios/README.md) for copy/paste commands, model guidance, RAG patterns, batch processing, and offline usage.

---

## 16. Notebooks

Notebooks are analysis clients, not the implementation layer:

```text
00_clip_foundations.ipynb
01_qwen3_vl_quickstart.ipynb
02_visual_document_understanding.ipynb
03_multimodal_rag.ipynb
```

Reusable logic belongs under `src/vlm_engineering/` and can be used without Jupyter.

---

## 17. Docker and Docker Compose

The repository includes `Dockerfile` and `docker-compose.yml` for reproducible CLI/container workflows.

The normal image does not bake model weights into the container.

### Build

```bash
docker compose build
```

### CLI

```bash
docker compose run --rm vlm-lab --help
docker compose run --rm vlm-lab models
```

### Mounted directories

```text
Host                       Container
./data                     /app/data
./models                   /app/models
hf-cache named volume      /home/appuser/.cache/huggingface
```

Explicit model directories under `./models` therefore survive one-shot container removal.

### Generative inference

```bash
docker compose run --rm vlm-lab \
  describe data/diagram_random_clean.png \
  --model-size 2b
```

### Download a local model

```bash
docker compose run --rm vlm-lab \
  download-model \
  --model-size 2b \
  --output models/Qwen3-VL-2B-Instruct
```

### Offline inference

```bash
docker compose run --rm \
  -e HF_HUB_OFFLINE=1 \
  vlm-lab \
  describe data/diagram_random_clean.png \
  --model-path models/Qwen3-VL-2B-Instruct
```

### Retrieval model downloads

```bash
docker compose run --rm vlm-lab \
  download-model \
  --embedding-size 2b \
  --output models/Qwen3-VL-Embedding-2B
```

```bash
docker compose run --rm vlm-lab \
  download-model \
  --reranker-size 2b \
  --output models/Qwen3-VL-Reranker-2B
```

### Hugging Face authentication

Pass credentials at runtime rather than baking them into the image:

```bash
export HF_TOKEN="..."

docker compose run --rm \
  -e HF_TOKEN \
  vlm-lab \
  describe data/diagram_random_clean.png \
  --model-id your-org/your-compatible-model
```

Do not commit tokens.

### Shell debugging

```bash
docker compose run --rm \
  --entrypoint /bin/sh \
  vlm-lab
```

### Direct Docker usage

```bash
docker build \
  -t vision-language-engineering-lab:0.3.1 \
  .
```

```bash
docker run --rm \
  vision-language-engineering-lab:0.3.1 \
  models
```

### GPU execution

The checked-in Docker setup is a portable baseline and does not force a particular GPU runtime.

Configure the appropriate container GPU runtime on the host and verify GPU visibility before relying on Qwen workloads.

---

## 18. Quality gates and compatibility validation

Before pushing changes:

```bash
python -m pip check
python -m ruff check .
python -m mypy .
python -m pytest \
  -m "not real_model" \
  --cov=vlm_engineering \
  --cov-report=term-missing
python -m pip_audit .
```

The project uses:

| Tool | Purpose |
|---|---|
| Ruff | linting, import organization, source-quality rules |
| mypy | static type checking |
| pytest | functional, integration, and contract testing |
| pytest-cov | coverage enforcement |
| `pip check` | installed dependency consistency |
| `pip-audit` | known dependency-vulnerability auditing |

For the v0.3.1 release branch, deterministic validation reached:

```text
340 passed
2 deselected
94.82% coverage
pip check: no broken requirements
pip-audit: no known vulnerabilities
```

The deselected tests are resource-heavy / opt-in real-model validations rather than normal CI tests.

CI validates Python 3.11, 3.12, and 3.13 while avoiding automatic multi-gigabyte model downloads in ordinary pull requests.

---

## 19. Troubleshooting

### `ModuleNotFoundError`

Activate the environment and install the appropriate extra:

```bash
source .venv/bin/activate
python -m pip install -e ".[all]"
```

### `vlm-lab: command not found`

```bash
python -m pip show vision-language-engineering-lab
which python
```

Then reinstall:

```bash
python -m pip install -e ".[all]"
```

### Missing Qwen retrieval dependencies

```bash
python -m pip install -e ".[qwen-retrieval]"
```

### Missing `torchvision`

Install the relevant package extra rather than adding ad-hoc dependencies individually:

```bash
python -m pip install -e ".[qwen]"
```

or:

```bash
python -m pip install -e ".[qwen-retrieval]"
```

### CUDA / RAM / out-of-memory error

Start with 2B models and batch size 1.

Reduce:

- batch size
- image resolution
- number of simultaneous images
- output token length

or move to hardware with more RAM/VRAM.

### Model offloading is slow

Automatic device placement can offload weights to CPU or disk on constrained machines. This is expected to be slower than running fully on an adequately sized GPU.

### First run is slow

The Hub may be downloading model weights.

For predictable storage:

```bash
vlm-lab download-model \
  --model-size 2b
```

Then use `--model-path`.

### Fully offline inference

Download the model first, then use a local model path and optionally:

```bash
export HF_HUB_OFFLINE=1
```

### Invalid/corrupt image

Local image validation intentionally fails early when possible.

Verify:

```bash
file path/to/image.png
```

and re-export the source image if necessary.

### Hugging Face authentication

For gated/private models, set:

```bash
export HF_TOKEN="..."
```

Do not commit credentials.

### Invalid YAML configuration

Validate before starting models:

```bash
vlm-lab validate-config \
  configs/qwen_multimodal_rag.example.yaml
```

### Dependency compatibility

```bash
python -m pip check
python -m pip show \
  sentence-transformers \
  torch \
  torchvision \
  transformers
```

### Which Qwen model should I choose?

Start with the 2B preset while validating the pipeline.

Move to larger models only when evaluation data demonstrates that the quality gain justifies the additional hardware cost.

---

## 20. Contributing

**Contributions are welcome.**

The project is intended to grow as an open Vision-Language engineering lab.

Useful contribution areas include:

- VLM integrations
- multimodal retrieval strategies
- document-understanding pipelines
- RAG evaluation
- persistent vector-database adapters
- performance and memory improvements
- deterministic and real-model testing
- Docker/deployment improvements
- documentation
- application scenarios
- bug fixes

### Contribution workflow

1. Fork or clone the repository.
2. Create a focused branch.
3. Implement the change.
4. Add/update tests.
5. Run quality gates.
6. Push the branch.
7. Open a Pull Request.

Example:

```bash
git clone https://github.com/helmighanmi/vision-language-engineering-lab.git
cd vision-language-engineering-lab

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[all,dev]"
```

Validation:

```bash
python -m pip check
python -m ruff check .
python -m mypy .
python -m pytest \
  -m "not real_model" \
  --cov=vlm_engineering \
  --cov-report=term-missing
python -m pip_audit .
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the complete contribution workflow.

Questions, feature proposals, documentation improvements, and bug reports are welcome through GitHub Issues.

---

## 21. Documentation

- `docs/model-loading.md` — model presets, Hub/custom/local modes and troubleshooting
- `docs/qwen-model-selection.md` — model-size selection and resource guidance
- `docs/rag-patterns.md` — visual and multimodal RAG patterns
- `docs/multimodal-retrieval.md` — Qwen embedding/reranking design and resource guidance
- `docs/architecture.md` — package architecture
- `docs/testing.md` — deterministic and real-model testing
- `docs/pdf/Vision_Language_Engineering_EN.pdf` — English teaching course
- `docs/pdf/Ingenierie_Vision_Langage_FR.pdf` — French teaching course

The README and Markdown documentation are the operational source of truth for the current package API/CLI.

---

## 22. Repository structure

```text
vision-language-engineering-lab/
├── .github/                 CI, security, and publishing workflows
├── configs/                 validated runtime YAML profiles
├── data/                    smoke-test/user images
├── docs/                    architecture, retrieval, RAG, testing guidance
├── examples/                small runnable API examples
├── models/                  explicit local model downloads (gitignored)
├── notebooks/               analysis/demonstration clients
├── scenarios/               end-to-end application cookbook
├── src/
│   └── vlm_engineering/     production Python package
├── tests/
│   ├── contract/            metadata/dependency/public-contract tests
│   ├── e2e/                 opt-in real-model tests
│   └── unit/                deterministic package tests
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE
```

---

## 23. Implemented vs roadmap

### Implemented

- CLIP foundations
- Qwen3-VL 2B / 4B / 8B generation
- image description and visual question answering
- structured visual/document analysis
- traceable visual chunks
- text-only visual RAG
- Qwen3-VL multimodal embeddings
- Qwen3-VL multimodal reranking
- two-stage multimodal retrieval
- multimodal RAG with original visual evidence
- project-local model downloads
- local/offline execution
- runtime YAML validation
- CLI workflows
- Docker workflows
- Python 3.11–3.13 CI
- deterministic tests and opt-in real-model tests

### Roadmap

Potential future directions include:

- persistent vector database adapters
- retrieval/RAG evaluation and benchmarks
- multi-image and richer multimodal sessions
- video / temporal reasoning
- richer document parsing and OCR workflows
- visual grounding / bounding boxes
- quantized / low-resource deployment
- async and streaming interfaces
- serving/API layer
- observability and caching
- visual-agent perception workflows

Roadmap items are not part of the current public API until implemented and validated.

---

## License and third-party material

Project code is distributed under Apache-2.0.

Model weights are downloaded separately and retain their respective licenses. Users are responsible for reviewing the license terms of third-party models, datasets, and external assets used with this package.
