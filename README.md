<!--
Path: README.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Vision-Language Engineering Lab

**CLIP, configurable Qwen3-VL, visual document understanding, semantic visual chunking, and multimodal RAG.**

`vision-language-engineering-lab` is a production-oriented learning and engineering package for building Vision-Language applications with CLIP, Qwen3-VL, document understanding, retrieval, reranking, and multimodal RAG.

The reusable implementation lives under `src/vlm_engineering`. Notebooks are analysis/demo clients only, while `examples/` and `scenarios/` provide runnable package usage patterns.

> **v0.2.1 compatibility target:** Python 3.11, 3.12, and 3.13. Compatibility is validated in CI on every supported Python version before release.

Contributions are welcome. See [Contributing](#20-contributing) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## 1. Quick start

### Install from the repository

For the current development version:

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

After a public PyPI release, users can install the package with:

```bash
python -m pip install "vision-language-engineering-lab[all]"
```

### Example 1 — Real-image smoke test with the default Qwen3-VL 2B model

A repository clone includes `data/diagram_random_clean.png` as a real-image smoke-test asset. Run:

```bash
vlm-lab describe data/diagram_random_clean.png \
  --model-size 2b \
  --prompt "Describe this image accurately. List the main objects, text, and important visual relationships."
```

The first Hub-backed run may download several gigabytes of model weights. Later runs reuse the Hugging Face cache. For predictable local/offline and Docker deployments, use the project-local model workflow in [Hugging Face cache and project-local models](#7-hugging-face-cache-and-project-local-models).

The equivalent Python API is:

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel(model_size="2b")

answer = model.generate(
    "data/diagram_random_clean.png",
    "Describe this image accurately. List the main objects, text, and important visual relationships.",
)

print(answer)
```

The default model is:

```text
Qwen/Qwen3-VL-2B-Instruct
```

If you installed the package from PyPI instead of cloning the repository, use your own image path:

```bash
vlm-lab describe /path/to/your/image.png \
  --model-size 2b \
  --prompt "Describe this image accurately. List the main objects, text, and important visual relationships."
```

### Example 2 — Select the 4B or 8B Qwen3-VL model

```python
from vlm_engineering import QwenVLModel

model_4b = QwenVLModel(model_size="4b")
model_8b = QwenVLModel(model_size="8b")

print(model_4b.model_source)
print(model_8b.model_source)
```

Available presets:

| Preset | Hugging Face model | Size class | Typical use |
|---|---|---:|---|
| `2b` | `Qwen/Qwen3-VL-2B-Instruct` | ~2B parameters | Default, learning, local prototypes |
| `4b` | `Qwen/Qwen3-VL-4B-Instruct` | ~4B parameters | Balanced quality/resource trade-off |
| `8b` | `Qwen/Qwen3-VL-8B-Instruct` | ~8B parameters | Higher-capacity inference on stronger hardware |

### Example 3 — Use an explicit compatible Hugging Face model

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel.from_hub(
    "Qwen/Qwen3-VL-4B-Instruct"
)

print(
    model.generate(
        "data/architecture.png",
        "Explain the architecture and the relationships between components.",
    )
)
```

### Example 4 — Load a model locally for offline inference

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel.from_local(
    "models/Qwen3-VL-4B-Instruct"
)

answer = model.generate(
    "data/architecture.png",
    "Describe the diagram using only the visual evidence.",
)

print(answer)
```

`from_local()` sets local-only loading and does not silently fall back to the Hugging Face Hub.

### Example 5 — Structured diagram/document analysis

```python
from vlm_engineering import QwenVLModel
from vlm_engineering.documents import analyze_visual_document

model = QwenVLModel(model_size="4b")
analysis = analyze_visual_document(model, "data/architecture.png")

print(analysis.to_dict())
```

The structured result can contain page type, title, summary, entities, relations, important text, and uncertainties.

### Example 6 — Text embedding for visual RAG

A common pattern is to use a VLM to convert visual evidence into faithful text and then index that text with Sentence Transformers.

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

### Example 7 — CLI

```bash
# List Qwen presets
vlm-lab models

# Default 2B
vlm-lab describe data/diagram_random_clean.png

# 4B
vlm-lab describe data/diagram_random_clean.png --model-size 4b

# 8B
vlm-lab analyze data/architecture.png --model-size 8b

# Explicit Hugging Face model
vlm-lab describe data/diagram_random_clean.png \
  --model-id Qwen/Qwen3-VL-4B-Instruct

# Local/offline model
vlm-lab describe data/diagram_random_clean.png \
  --model-path models/Qwen3-VL-4B-Instruct
```

For complete end-to-end workflows, see [`scenarios/README.md`](scenarios/README.md).

---

## 2. What this project demonstrates

```text
CLIP foundations
   -> image/text embeddings and zero-shot similarity

Qwen3-VL Instruct (2B / 4B / 8B or custom model ID)
   -> image description, VQA, diagram/table/screenshot understanding, structured JSON

Document understanding
   -> native text + visual semantics -> traceable chunks

Visual RAG
   -> VLM descriptions + normal text embedder

Multimodal RAG
   -> Qwen3-VL-Embedding -> retrieval -> Qwen3-VL-Reranker -> Qwen3-VL answer
```

The project intentionally separates:

- reusable package code under `src/vlm_engineering/`
- deterministic tests under `tests/`
- small API examples under `examples/`
- end-to-end application recipes under `scenarios/`
- exploratory notebooks under `notebooks/`
- architecture and operational documentation under `docs/`

---

## 3. Python compatibility

v0.2.1 targets:

| Python | Support |
|---|---:|
| 3.11 | ✅ Supported |
| 3.12 | ✅ Supported |
| 3.13 | ✅ Supported |

The package metadata declares:

```text
>=3.11,<3.14
```

GitHub Actions validates the supported versions independently. The compatibility matrix checks package installation, dependency resolution, runtime imports, Ruff, mypy, pytest, coverage, and dependency security auditing.

Ruff targets Python 3.11 syntax so package code does not accidentally rely on Python 3.12+ syntax.

### Google Colab

Google Colab can use the package directly when its runtime Python version is within the supported range.

After the public v0.2.1 release:

```python
!pip install "vision-language-engineering-lab[all]"
```

Then:

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel(model_size="2b")
print(model.model_source)
```

For GPU workloads, verify the Colab runtime has a GPU enabled before loading large model weights.

---

## 4. Installation options

### Requirements

- Python **3.11, 3.12, or 3.13**
- Git for source installs
- Enough disk space for the model(s) you choose to download
- A GPU is strongly recommended for Qwen3-VL inference
- CLIP and lightweight package tests can run on CPU

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

Retrieval / Sentence Transformers:

```bash
python -m pip install -e ".[retrieval]"
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

The default generative VLM is **`Qwen/Qwen3-VL-2B-Instruct`**. The package also provides friendly 4B and 8B presets and supports explicit compatible Hugging Face model IDs.

| Preset | Hugging Face model | Size class | Recommended use |
|---|---|---:|---|
| `2b` | `Qwen/Qwen3-VL-2B-Instruct` | ~2B parameters | **Default.** Learning, local prototypes, lower resource use |
| `4b` | `Qwen/Qwen3-VL-4B-Instruct` | ~4B parameters | Balanced quality/resource option |
| `8b` | `Qwen/Qwen3-VL-8B-Instruct` | ~8B parameters | Higher-capacity inference on stronger hardware |

Official model pages:

- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

The size labels are parameter classes, not exact runtime-memory requirements. A rough BF16 weight-only floor is about 2 bytes per parameter (~4 GB for 2B, ~8 GB for 4B, ~16 GB for 8B). Real inference needs additional memory for the vision encoder, activations, KV cache, image tokens, framework overhead, and generation state.

Other Qwen models in the retrieval stack:

| Model | Role |
|---|---|
| `Qwen/Qwen3-VL-Embedding-2B` | Multimodal retrieval embeddings |
| `Qwen/Qwen3-VL-Reranker-2B` | Second-stage candidate reranking |

Qwen3-VL is supported through current Hugging Face Transformers. `trust_remote_code` is disabled by default and is not required for the normal Qwen3-VL path implemented by this package.

---

## 6. Model selection and loading

### Default 2B

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel()
```

### 4B or 8B preset

```python
model_4b = QwenVLModel(model_size="4b")
model_8b = QwenVLModel.from_preset("8b")
```

### Explicit Hugging Face model ID

```python
model = QwenVLModel.from_hub(
    "Qwen/Qwen3-VL-4B-Instruct"
)
```

### Explicit local model

```python
model = QwenVLModel.from_local(
    "models/Qwen3-VL-4B-Instruct"
)
```

CLI model-selection flags are mutually exclusive:

```text
--model-size
--model-id
--model-path
```

This prevents the CLI from silently selecting a different source than the user intended.

### Programmatic registry access

```python
from vlm_engineering import QWEN3_VL_INSTRUCT_MODELS

for alias, preset in QWEN3_VL_INSTRUCT_MODELS.items():
    print(alias, preset.model_id, preset.parameter_class)
```

---

## 7. Hugging Face cache and project-local models

### Two supported storage modes

For a quick experiment, Hub-backed inference can use the normal Hugging Face cache:

```bash
vlm-lab describe data/diagram_random_clean.png --model-size 2b
```

For Docker, offline execution, explicit cleanup, and reproducible deployments, prefer an explicit project-local model under `models/`.

### Recommended: download the 2B preset into `models/`

```bash
vlm-lab download-model \
  --model-size 2b \
  --output models/Qwen3-VL-2B-Instruct
```

The `--output` argument is optional. This shorter command uses the same deterministic destination:

```bash
vlm-lab download-model --model-size 2b
```

Default destination:

```text
models/Qwen3-VL-2B-Instruct/
```

Hugging Face `snapshot_download(..., local_dir=...)` places the model files under the requested directory instead of the normal global Hub cache. Hugging Face may create a small `.cache/huggingface/` metadata directory inside the local model directory.

### Run from the explicit local model directory

```bash
vlm-lab describe data/diagram_random_clean.png \
  --model-path models/Qwen3-VL-2B-Instruct \
  --prompt "Describe this image accurately. List the main objects, text, and important visual relationships."
```

### Verify fully offline loading

After the download succeeds:

```bash
HF_HUB_OFFLINE=1 vlm-lab describe data/diagram_random_clean.png \
  --model-path models/Qwen3-VL-2B-Instruct \
  --prompt "Describe this image accurately."
```

`QwenVLModel.from_local()` enables local-files-only loading and does not silently fall back to the Hub.

### Download another preset or explicit model ID

```bash
vlm-lab download-model --model-size 4b
vlm-lab download-model --model-size 8b
```

```bash
vlm-lab download-model \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --output models/qwen8b
```

### Clean the old global cache only after offline verification

Never remove the global Hugging Face cache while a model is downloading or running. First verify the local model with `HF_HUB_OFFLINE=1`, then inspect the old cache:

```bash
du -sh ~/.cache/huggingface/hub 2>/dev/null
hf cache list --sort size:desc
```

If disk space is needed after the offline test succeeds, remove only the old Qwen cache entry you no longer need:

```bash
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct
```

The package never deletes a user's shared Hugging Face cache automatically.

---

## 8. Advanced model-loading options

### `trust_remote_code`

Default: **False**.

```bash
vlm-lab describe image.jpg --trust-remote-code
```

Only enable this for a model repository you trust and that genuinely requires repository-provided Python code. Enabling it expands the security trust boundary.

### Device and dtype

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel(
    model_size="4b",
    device_map="auto",
    dtype="auto",
)
```

Validate the exact dtype, quantization strategy, backend, and model size on your target hardware before production deployment.

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
  --prompt "Which metric increased the most and what evidence supports it?"
```

### Diagram / architecture understanding

```bash
vlm-lab analyze architecture.png --model-size 4b
```

The `analyze` command asks the VLM for structured JSON containing page type, title, summary, entities, relations, important text, and uncertainties.

### Screenshot / UI understanding

Use `describe` or `analyze` for dashboards, application screenshots, forms, and technical UI states.

### Tables and document pages

Render the page as an image and request structured extraction. For production documents, preserve native/OCR text separately and use the VLM for layout, grouping, relationships, and visual semantics.

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

A VLM can replace OCR in some workflows, but the strongest document pipeline usually keeps native/OCR evidence for literal accuracy and adds VLM semantics where visual structure matters.

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
   -> Qwen3-VL description / structured analysis
   -> retrieval text
   -> Sentence Transformers text embedder
   -> vector index / vector database
```

This is useful when an existing RAG platform only supports text embeddings. The VLM turns visual evidence into searchable text while the original `image_ref` is retained for final grounding.

```bash
python examples/text_only_visual_rag.py
```

After retrieval, the final VLM can inspect the original image again rather than trusting only the generated description.

---

## 13. True multimodal embeddings and reranking

### Multimodal embeddings

```python
from vlm_engineering.retrieval import QwenMultimodalEmbedder

embedder = QwenMultimodalEmbedder(
    "Qwen/Qwen3-VL-Embedding-2B"
)

queries = [
    "Find the architecture diagram containing a cache and database."
]

documents = [
    "A textual architecture note",
    "data/architecture.png",
    {"text": "Payment architecture", "image": "data/architecture.png"},
]

q = embedder.encode(
    queries,
    prompt="Retrieve relevant technical document evidence.",
)
d = embedder.encode(documents)

print(q @ d.T)
```

### Multimodal reranking

```python
from vlm_engineering.retrieval import QwenMultimodalReranker

reranker = QwenMultimodalReranker()

scores = reranker.score(
    "Which diagram shows the payment database?",
    ["candidate text", "data/diagram.png"],
)

print(scores)
```

Recommended production pattern:

```text
query
  -> Qwen3-VL-Embedding -> top K candidates
  -> Qwen3-VL-Reranker  -> top 3-5
  -> retrieve chunk text + original image
  -> selected Qwen3-VL-Instruct model
  -> grounded answer with source/page
```

---

## 14. CLIP foundations

```python
from PIL import Image

from vlm_engineering import CLIPEncoder

image = Image.open("data/diagram_random_clean.png").convert("RGB")

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
examples/clip_zero_shot.py                 CLIP zero-shot classification
examples/qwen_describe_image.py            Qwen image description
examples/qwen_model_selection.py           2B/4B/8B/custom model selection
examples/qwen_local_model.py               explicit local/offline Qwen
examples/structured_visual_chunk.py        RAG-ready visual chunk
examples/text_only_visual_rag.py           VLM description + text embeddings
examples/qwen_multimodal_retrieval.py      multimodal embedding/retrieval
```

End-to-end application cookbook:

```text
scenarios/scenario_01_image_captioning.py              image captioning / description
scenarios/scenario_02_visual_question_answering.py     grounded VQA
scenarios/scenario_03_diagram_to_json.py               diagram -> structured JSON
scenarios/scenario_04_document_page_to_rag_chunk.py    native/OCR + VLM -> RAG chunk
scenarios/scenario_05_text_only_visual_rag.py          images -> VLM text -> text RAG
scenarios/scenario_06_true_multimodal_retrieval.py     direct multimodal retrieval
scenarios/scenario_07_compare_qwen_presets.py          compare Qwen 2B / 4B / 8B
scenarios/scenario_08_hub_vs_local_loading.py          Hub/cache vs local/offline
scenarios/scenario_09_batch_structured_analysis.py     batch image/page analysis
```

Each applicable scenario accepts the same model choices used by the package:

```bash
# Default 2B
python scenarios/scenario_01_image_captioning.py data/diagram_random_clean.png

# 4B / 8B
python scenarios/scenario_01_image_captioning.py \
  data/diagram_random_clean.png --model-size 4b

python scenarios/scenario_01_image_captioning.py \
  data/diagram_random_clean.png --model-size 8b

# Explicit Hugging Face model
python scenarios/scenario_01_image_captioning.py \
  data/diagram_random_clean.png \
  --model-id Qwen/Qwen3-VL-4B-Instruct

# Explicit local model
python scenarios/scenario_01_image_captioning.py \
  data/diagram_random_clean.png \
  --model-path models/Qwen3-VL-4B-Instruct
```

See [`scenarios/README.md`](scenarios/README.md) for copy/paste commands, model guidance, RAG patterns, batch processing, and offline usage.

---

## 16. Notebooks

Notebooks are **analysis clients**, not the implementation layer:

- `00_clip_foundations.ipynb`
- `01_qwen3_vl_quickstart.ipynb`
- `02_visual_document_understanding.ipynb`
- `03_multimodal_rag.ipynb`

All reusable logic lives under `src/vlm_engineering` and can be used without Jupyter.

---

## 17. Docker and Docker Compose

The repository includes both a `Dockerfile` and `docker-compose.yml` so users can run the CLI inside a reproducible container instead of managing the Python environment directly on the host.

The current container is a runtime-oriented image. It installs the package with the Qwen and retrieval dependencies and does not include contributor/test/notebook tooling by default.

### Prerequisites

Install one of:

- Docker Desktop, or
- Docker Engine with Docker Compose v2

Verify:

```bash
docker --version
docker compose version
```

### Build with Docker Compose

From the repository root:

```bash
docker compose build
```

The image is built from the repository `Dockerfile`, which currently uses Python 3.12 as the reference container runtime. The package itself targets Python 3.11-3.13 in v0.2.1.

### Check the CLI inside the container

```bash
docker compose run --rm vlm-lab --help
```

List the available Qwen presets:

```bash
docker compose run --rm vlm-lab models
```

### Understand the mounted directories

The Compose configuration mounts:

```text
Host                       Container
./data                     /app/data                 read-only
./models                   /app/models               persistent bind mount
hf-cache named volume      /home/appuser/.cache/huggingface
```

This means:

- put user images/pages under `./data/`
- explicitly downloaded models under `./models/` survive container removal
- Hugging Face cache downloads survive normal `docker compose run --rm` commands
- the container does not need to copy your private data into the image

### Describe an image with the default 2B model

Place an image at `data/diagram_random_clean.png`, then run:

```bash
docker compose run --rm vlm-lab \
  describe data/diagram_random_clean.png
```

With a custom prompt:

```bash
docker compose run --rm vlm-lab \
  describe data/diagram_random_clean.png \
  --prompt "Describe this image precisely."
```

### Select 4B or 8B inside the container

```bash
docker compose run --rm vlm-lab \
  describe data/diagram_random_clean.png \
  --model-size 4b
```

```bash
docker compose run --rm vlm-lab \
  analyze data/architecture.png \
  --model-size 8b
```

### Persist a model explicitly under `./models`

Download the 4B preset from inside the container:

```bash
docker compose run --rm vlm-lab \
  download-model \
  --model-size 4b \
  --output models/Qwen3-VL-4B-Instruct
```

Because `/app/models` is bind-mounted to host `./models`, the downloaded files remain available after the container exits.

Then use the local model explicitly:

```bash
docker compose run --rm vlm-lab \
  describe data/diagram_random_clean.png \
  --model-path models/Qwen3-VL-4B-Instruct
```

### Fully offline container inference

First download the model while network access is available. Then run:

```bash
docker compose run --rm \
  -e HF_HUB_OFFLINE=1 \
  vlm-lab \
  describe data/diagram_random_clean.png \
  --model-path models/Qwen3-VL-4B-Instruct
```

`--model-path` enables local-only model loading in the package.

### Hugging Face authentication without baking secrets into the image

For a private or gated compatible repository, keep the token on the host and pass it at runtime rather than storing it in the Dockerfile:

```bash
export HF_TOKEN="..."

docker compose run --rm \
  -e HF_TOKEN \
  vlm-lab \
  describe data/diagram_random_clean.png \
  --model-id your-org/your-compatible-model
```

Do not commit Hugging Face tokens or other credentials to the repository.

### Open a shell for debugging

The image entrypoint is `vlm-lab`. To open a shell instead:

```bash
docker compose run --rm \
  --entrypoint /bin/sh \
  vlm-lab
```

Inside the container you can inspect:

```bash
python --version
python -m pip list
ls -la /app/data
ls -la /app/models
```

### Build and run without Compose

Build:

```bash
docker build \
  -t vision-language-engineering-lab:0.2.1 \
  .
```

List models:

```bash
docker run --rm \
  vision-language-engineering-lab:0.2.1 \
  models
```

Run against host data and persistent models:

```bash
docker run --rm \
  -v "$PWD/data:/app/data:ro" \
  -v "$PWD/models:/app/models" \
  -v vlm-hf-cache:/home/appuser/.cache/huggingface \
  vision-language-engineering-lab:0.2.1 \
  describe data/diagram_random_clean.png \
  --model-size 2b
```

### Rebuild after package/dependency changes

```bash
docker compose build
```

If you specifically need a clean image rebuild:

```bash
docker compose build --no-cache
```

### Remove containers while keeping explicit local models

```bash
docker compose down
```

The host `./models` directory remains because it is a bind mount.

To also delete the named Hugging Face cache volume:

```bash
docker compose down -v
```

Use `-v` carefully because cached model downloads in the named volume will be removed.

### GPU execution

The checked-in Compose file is intentionally a portable baseline and does not force a GPU runtime configuration.

For NVIDIA GPU inference, configure a compatible host Docker/NVIDIA runtime and verify GPU visibility from the container before relying on it for Qwen workloads. GPU setup is host-specific; the project does not assume every user has NVIDIA hardware.

Qwen3-VL 4B and especially 8B require substantially more memory than the default 2B model. Start with 2B when validating a container deployment.

### Why `docker compose run` instead of `docker compose up`?

`vlm-lab` is a command-line workload, not a long-running web server. The Compose service defaults to `--help`, so most users should execute individual tasks with:

```bash
docker compose run --rm vlm-lab <command> ...
```

This creates an isolated one-shot container, runs the command, and removes the container afterward while preserving the mounted data/model/cache volumes.

---

## 18. Quality gates and compatibility validation

Before pushing changes:

```bash
python -m pip check
python -m ruff check .
python -m mypy .
python -m pytest --cov=vlm_engineering --cov-report=term-missing
python -m pip_audit
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

CI additionally executes the package against Python 3.11, 3.12, and 3.13.

Heavy ML components are mocked or injected in normal unit tests, so CI does not need to download multi-gigabyte Qwen model weights merely to validate package contracts.

---

## 19. Troubleshooting

### `ModuleNotFoundError`

Activate the environment and install the required extra:

```bash
source .venv/bin/activate
python -m pip install -e ".[all]"
```

### `vlm-lab: command not found`

Confirm the package is installed in the active environment:

```bash
python -m pip show vision-language-engineering-lab
which python
```

Then reinstall:

```bash
python -m pip install -e ".[all]"
```

### CUDA / out-of-memory error

Start with the default `2b` preset, reduce image resolution/output length, or use stronger hardware. Moving from 2B to 4B to 8B increases model capacity and memory pressure.

### First run is slow

Hub/cache mode may be downloading model files. For a predictable deployment directory, use `vlm-lab download-model --model-size 2b`, then run with `--model-path models/Qwen3-VL-2B-Instruct`.

### Fully offline inference

Download the model first, then use `--model-path` / `QwenVLModel.from_local()`. Optionally set:

```bash
export HF_HUB_OFFLINE=1
```

### Model requires repository-provided Python code

Do not enable `--trust-remote-code` automatically. Verify the model repository and only opt in when genuinely required.

### Sentence Transformers / dependency compatibility

Check the resolved environment:

```bash
python -m pip check
python -m pip show sentence-transformers torch transformers
```

v0.2.1 allows Sentence Transformers 5.4 through the 6.x line, while CI validates the dependency stack on Python 3.11-3.13.

### Which Qwen model should I choose?

- Start with **2B** while developing the pipeline.
- Compare **4B** when reasoning/extraction quality is insufficient.
- Test **8B** when the quality gain justifies the higher resource cost.
- Evaluate on your own diagrams/documents; model size alone does not guarantee better RAG accuracy.

### Docker cannot see my file

Confirm the file is under the host `data/` directory because Compose mounts that directory into `/app/data`:

```bash
ls -la data/
docker compose run --rm --entrypoint /bin/sh vlm-lab -c "ls -la /app/data"
```

### Docker keeps downloading models

For production-like Docker tests, prefer downloading the model explicitly into `./models` and using `--model-path`. The named Hugging Face cache volume remains useful for ad-hoc Hub-backed experiments.

---

## 20. Contributing

**Contributions are welcome.**

The project is intended to grow as an open Vision-Language engineering lab, and contributions from developers, AI/ML engineers, researchers, students, and practitioners are encouraged.

Useful contributions include:

- new VLM integrations
- additional Qwen-compatible model support
- multimodal retrieval strategies
- document-understanding pipelines
- RAG evaluation methods
- performance and memory improvements
- additional deterministic tests
- Python-version compatibility improvements
- Docker/deployment improvements
- documentation and tutorials
- reproducible application scenarios
- bug fixes

### Contribution workflow

1. Fork the repository.
2. Create a focused branch.
3. Implement the change.
4. Add or update tests.
5. Run the quality gates locally.
6. Push the branch.
7. Open a Pull Request describing the motivation, implementation, validation, and any compatibility considerations.

Example development setup:

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
python -m pytest --cov=vlm_engineering --cov-report=term-missing
python -m pip_audit
```

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the complete contribution workflow.

Questions, feature proposals, documentation improvements, and bug reports are welcome through GitHub Issues.

---

## 21. Documentation

- `docs/model-loading.md` — model presets, Hub/custom/local modes and troubleshooting
- `docs/qwen-model-selection.md` — model-size selection and memory guidance
- `docs/rag-patterns.md` — visual and multimodal RAG patterns
- `docs/architecture.md` — package architecture
- `docs/testing.md` — testing strategy
- `docs/pdf/Vision_Language_Engineering_EN.pdf` — English teaching course
- `docs/pdf/Ingenierie_Vision_Langage_FR.pdf` — French teaching course

The PDFs are conceptual teaching references. The README and Markdown docs are the operational source of truth for the current package CLI/API.

---

## 22. Repository structure

```text
vision-language-engineering-lab/
├── .github/                 CI, security, and publishing workflows
├── data/                    user-provided images/pages (gitignored except docs)
├── docs/                    architecture, model-loading, RAG, testing guidance
├── examples/                small runnable API examples
├── models/                  explicit local model downloads (gitignored)
├── notebooks/               analysis/demonstration clients
├── scenarios/               end-to-end application cookbook
├── src/
│   └── vlm_engineering/     production Python package
├── tests/                   deterministic unit/integration/contract tests
├── Dockerfile               container image definition
├── docker-compose.yml       reproducible CLI/container workflow
├── pyproject.toml           package metadata and dependency source of truth
├── requirements.txt         compatibility installer
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE
```

---

## License and third-party material

Project code is distributed under Apache-2.0. Model weights are downloaded separately and retain their respective licenses. Users are responsible for reviewing the license terms of any third-party models, datasets, or external assets they use with this package.
