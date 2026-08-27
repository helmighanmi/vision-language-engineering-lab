<!--
Path: README.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Vision-Language Engineering Lab

**CLIP, configurable Qwen3-VL, visual document understanding, semantic visual chunking and multimodal RAG.**

This repository is a production-oriented learning and engineering lab that moves from classic CLIP embeddings to modern generative Vision-Language Models (VLMs), document understanding and multimodal retrieval.

The reusable implementation lives under `src/vlm_engineering`. Notebooks are analysis/demo clients only.

## What this project demonstrates

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

## Qwen models used by this project

The default generative VLM is **`Qwen/Qwen3-VL-2B-Instruct`**. The package also provides friendly presets for the 4B and 8B Instruct variants, plus an escape hatch for any compatible Hugging Face model ID.

| Preset | Hugging Face model | Size class | Recommended use |
|---|---|---:|---|
| `2b` | `Qwen/Qwen3-VL-2B-Instruct` | ~2B parameters | **Default.** Learning, local prototypes, lower resource use |
| `4b` | `Qwen/Qwen3-VL-4B-Instruct` | ~4B parameters | Balanced quality/resource option |
| `8b` | `Qwen/Qwen3-VL-8B-Instruct` | ~8B parameters | Higher-capacity inference on stronger hardware |

Official model pages:

- https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct

The size names are parameter classes, not exact memory requirements. A rough BF16 weight-only floor is about 2 bytes per parameter (~4 GB for 2B, ~8 GB for 4B, ~16 GB for 8B), but real inference needs additional memory for the vision encoder, activations, KV cache, image tokens and runtime overhead. Measure on your own workload before choosing a production size.

Other Qwen models in the retrieval stack:

| Model | Role |
|---|---|
| `Qwen/Qwen3-VL-Embedding-2B` | multimodal retrieval embeddings |
| `Qwen/Qwen3-VL-Reranker-2B` | second-stage candidate reranking |

> Qwen3-VL is natively integrated into current Hugging Face Transformers. `trust_remote_code` is disabled by default and is **not required** for the normal Qwen3-VL path implemented here.

---

## 1. Installation

### Requirements

- Python **3.12**
- Git
- Enough disk space for the model(s) you choose to download
- GPU strongly recommended for Qwen3-VL; CLIP can run on CPU for small experiments

### Create a virtual environment

Linux/macOS/GitHub Codespaces:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### Choose what to install

Development/tests only:

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

Multimodal embedding/reranking:

```bash
python -m pip install -e ".[retrieval]"
```

Full repository environment:

```bash
python -m pip install -e ".[all,dev,notebooks]"
```

A compatibility `requirements.txt` is also included for users who prefer requirements files:

```bash
python -m pip install -r requirements.txt
```

`pyproject.toml` remains the dependency source of truth.

Confirm the CLI:

```bash
vlm-lab --help
```

---

## 2. Quick start: choose your Qwen3-VL model

List the built-in presets:

```bash
vlm-lab models
```

### Default: 2B

No model flag is needed:

```bash
vlm-lab describe data/example.jpg \
  --prompt "Describe this image precisely."
```

Equivalent Python:

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel()  # Qwen/Qwen3-VL-2B-Instruct
answer = model.generate("data/example.jpg", "Describe this image precisely.")
print(answer)
```

### Use the 4B preset

CLI:

```bash
vlm-lab describe data/example.jpg \
  --model-size 4b \
  --prompt "Describe the image and identify important relationships."
```

Python:

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel(model_size="4b")
print(model.model_source)
# Qwen/Qwen3-VL-4B-Instruct
```

### Use the 8B preset

```bash
vlm-lab analyze data/architecture.png --model-size 8b
```

```python
model = QwenVLModel(model_size="8b")
```

### Use an explicit compatible Hugging Face model ID

This keeps the package extensible to future Qwen variants without waiting for a package release:

```bash
vlm-lab describe data/example.jpg \
  --model-id Qwen/Qwen3-VL-4B-Instruct
```

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel.from_hub("Qwen/Qwen3-VL-4B-Instruct")
```

`--model-size`, `--model-id` and `--model-path` are mutually exclusive so the CLI never silently chooses a different model than you intended.

---

## 3. Hugging Face cache vs explicit local/offline model

### Option A - Hub/cache mode

The first run downloads weights if needed. Later runs reuse the Hugging Face cache.

```bash
vlm-lab describe data/example.jpg --model-size 4b
```

Python:

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel.from_preset("4b")
```

### Option B - download a preset explicitly

```bash
vlm-lab download-model --model-size 4b
```

The default destination is derived from the model name:

```text
models/Qwen3-VL-4B-Instruct/
```

Choose your own destination if desired:

```bash
vlm-lab download-model \
  --model-size 4b \
  --output /models/qwen4b
```

### Option C - download any compatible model ID

```bash
vlm-lab download-model \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --output models/qwen8b
```

### Option D - explicit local/offline inference

```bash
vlm-lab describe data/example.jpg \
  --model-path models/Qwen3-VL-4B-Instruct \
  --prompt "Describe the image precisely."
```

Python:

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel.from_local("models/Qwen3-VL-4B-Instruct")
print(model.generate("data/example.jpg", "What is happening in this image?"))
```

`from_local()` sets `local_files_only=True`; it does not silently fall back to the Hugging Face Hub.

For a fully disconnected environment you can additionally set:

```bash
export HF_HUB_OFFLINE=1
```

---

## 4. Advanced model-loading options

### `trust_remote_code`

Default: **False**.

```bash
vlm-lab describe image.jpg --trust-remote-code
```

Only enable this for a model repository you trust and that genuinely requires repository-provided Python code. Enabling it expands the security trust boundary.

### Device and dtype from Python

```python
from vlm_engineering import QwenVLModel

model = QwenVLModel(
    model_size="4b",
    device_map="auto",
    dtype="auto",
)
```

For production, validate the exact dtype/quantization/backend combination on your hardware instead of assuming parameter count alone predicts memory usage.

---

## 5. Core Qwen3-VL use cases

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

The `analyze` command asks the VLM for stable JSON containing page type, title, summary, entities, relations, important text and uncertainties.

### Screenshot / UI understanding

Use `describe` or `analyze` for dashboards, application screenshots, forms and technical UI states.

### Tables and document pages

Render a page as an image and request structured extraction. For production documents, preserve native/OCR text separately and use the VLM for layout, grouping and relationships.

---

## 6. VLM + OCR/native parsing: hybrid, not replacement by default

```text
Native extraction / OCR                    Qwen3-VL
exact text, IDs, numbers              layout, arrows, groups,
long deterministic text               relationships, visual meaning
             \                         /
              \                       /
                    fusion
                      |
               RAG-ready chunk
```

A VLM can replace OCR in some workflows, but the strongest document pipeline usually keeps native/OCR evidence for literal accuracy and adds VLM semantics where visual structure matters.

---

## 7. Build a RAG-ready visual chunk

```bash
vlm-lab chunk data/page_001.png \
  --model-size 4b \
  --document-id architecture-v1 \
  --source-file architecture.pdf \
  --page 1 \
  --native-text-file data/page_001.txt
```

The result preserves traceability such as document ID, page, image reference, entities, relations and fused retrieval text.

---

## 8. Visual RAG when you only have a text embedder

```text
image/page -> Qwen3-VL description/JSON -> retrieval text -> text embedder -> vector DB
```

This is useful when an existing RAG platform only supports text embeddings. The VLM turns visual evidence into faithful searchable text.

```bash
python examples/text_only_visual_rag.py
```

Keep the original `image_ref`; after retrieval, the final VLM can inspect the original visual evidence again instead of trusting only the generated description.

---

## 9. True multimodal embeddings

```python
from vlm_engineering.retrieval import QwenMultimodalEmbedder

embedder = QwenMultimodalEmbedder("Qwen/Qwen3-VL-Embedding-2B")

queries = ["Find the architecture diagram containing a cache and database."]
documents = [
    "A textual architecture note",
    "data/architecture.png",
    {"text": "Payment architecture", "image": "data/architecture.png"},
]

q = embedder.encode(queries, prompt="Retrieve relevant technical document evidence.")
d = embedder.encode(documents)
print(q @ d.T)
```

---

## 10. Multimodal reranking

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
  -> Qwen3-VL-Embedding -> top 20
  -> Qwen3-VL-Reranker  -> top 3-5
  -> retrieve chunk text + original image
  -> selected Qwen3-VL-Instruct model
  -> grounded answer with source/page
```

---

## 11. CLIP foundations

```python
from PIL import Image
from vlm_engineering import CLIPEncoder

image = Image.open("data/example.jpg").convert("RGB")
clip = CLIPEncoder()
print(clip.zero_shot_classify(image, ["cat", "dog", "airplane"]))
```

CLIP is useful for learning the shared image/text embedding-space idea that modern multimodal retrieval builds upon.

---

## 12. Python model-selection API summary

```python
from vlm_engineering import QwenVLModel

# Default 2B
m1 = QwenVLModel()

# Friendly presets
m2 = QwenVLModel(model_size="4b")
m3 = QwenVLModel.from_preset("8b")

# Explicit compatible Hugging Face ID
m4 = QwenVLModel.from_hub("Qwen/Qwen3-VL-4B-Instruct")

# Explicit local/offline directory
m5 = QwenVLModel.from_local("models/Qwen3-VL-4B-Instruct")
```

Programmatic registry access:

```python
from vlm_engineering import QWEN3_VL_INSTRUCT_MODELS

for alias, preset in QWEN3_VL_INSTRUCT_MODELS.items():
    print(alias, preset.model_id, preset.parameter_class)
```

---

## 13. Examples

```text
examples/clip_zero_shot.py               CLIP zero-shot classification
examples/qwen_describe_image.py          Qwen image description
examples/qwen_model_selection.py         2B/4B/8B/custom model selection
examples/qwen_local_model.py             explicit local/offline Qwen
examples/structured_visual_chunk.py      RAG-ready visual chunk
examples/text_only_visual_rag.py         VLM description + text embeddings
examples/qwen_multimodal_retrieval.py    multimodal embedding/retrieval
```

---

## 14. Notebooks

Notebooks are **analysis clients**, not the implementation layer:

- `00_clip_foundations.ipynb`
- `01_qwen3_vl_quickstart.ipynb`
- `02_visual_document_understanding.ipynb`
- `03_multimodal_rag.ipynb`

All reusable logic lives under `src/vlm_engineering` and can be used without Jupyter.

---

## 15. Quality gates

Before pushing changes:

```bash
python -m ruff check .
python -m mypy .
python -m pytest
python -m pip_audit .
```

The test suite is intentionally lightweight and mocks/injects heavy ML components, so normal CI does not download multi-gigabyte Qwen/CLIP weights.

---

## 16. Docker

Build:

```bash
docker build -t vision-language-engineering-lab .
```

Run with your data/model cache mounted:

```bash
docker compose run --rm vlm-lab describe data/example.jpg
```

GPU execution depends on the host/container runtime. For heavier concurrent Qwen workloads, consider a dedicated inference backend/server rather than loading a separate large model per process.

---

## 17. Troubleshooting

### `ModuleNotFoundError`

Activate the project virtual environment and reinstall the appropriate optional dependencies:

```bash
source .venv/bin/activate
python -m pip install -e ".[all]"
```

### CUDA / out-of-memory error

Start with the default `2b` preset, reduce image resolution/output length, or use stronger hardware. Moving from 2B -> 4B -> 8B increases model capacity and memory pressure.

### First run is slow

Hub/cache mode may be downloading model files. Use `vlm-lab download-model ...` if you want an explicit download step before inference.

### Need fully offline inference

Download the model first, then use `--model-path` / `QwenVLModel.from_local()`. Optionally set `HF_HUB_OFFLINE=1`.

### Model requires custom repository code

Do not enable `--trust-remote-code` automatically. Verify the model repository and only opt in when required.

### Which model should I choose?

- Start with **2B** while developing the pipeline.
- Compare **4B** if visual reasoning/extraction quality is insufficient.
- Test **8B** only when the quality gain justifies higher resource cost.
- Evaluate on your own diagrams/documents; model size alone does not guarantee better RAG accuracy.

---

## 18. Documentation

- `docs/model-loading.md` - model presets, Hub/custom/local modes and troubleshooting
- `docs/qwen-model-selection.md` - model-size selection and memory guidance
- `docs/rag-patterns.md` - visual and multimodal RAG patterns
- `docs/architecture.md` - package architecture
- `docs/testing.md` - testing strategy
- `docs/pdf/Vision_Language_Engineering_EN.pdf` - English teaching course
- `docs/pdf/Ingenierie_Vision_Langage_FR.pdf` - French teaching course

The PDFs are conceptual teaching references. The README and Markdown docs are the operational source of truth for the current package CLI/API.

---

## 19. Repository structure

```text
src/vlm_engineering/     production package
examples/                runnable examples
notebooks/               analysis/demonstration clients
tests/                   deterministic unit/integration/contract tests
docs/                    architecture, model-loading and RAG guidance
legacy/                  previous educational notebooks isolated from maintained code
models/                  explicitly downloaded weights (gitignored)
data/                    user-provided images/pages (gitignored)
```

## License and third-party material

New maintained project code is distributed under Apache-2.0. Model weights are downloaded separately and retain their own licenses. Historical notebooks are isolated under `legacy/`; see `THIRD_PARTY_NOTICES.md` before redistributing them.
