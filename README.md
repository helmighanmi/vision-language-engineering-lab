<!--
Path: README.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Vision-Language Engineering Lab

**CLIP, Qwen3-VL, visual document understanding, semantic visual chunking and multimodal RAG.**

This repository is a production-oriented learning and engineering lab that moves from classic CLIP embeddings to modern generative Vision-Language Models and multimodal retrieval.

## What this project demonstrates

```text
CLIP foundations
   -> image/text embeddings and zero-shot similarity
Qwen3-VL-2B-Instruct
   -> image description, VQA, diagram/table/screenshot understanding, structured JSON
Document understanding
   -> native text + visual semantics -> traceable chunks
Visual RAG
   -> VLM descriptions + normal text embedder
Multimodal RAG
   -> Qwen3-VL-Embedding -> retrieval -> Qwen3-VL-Reranker -> Qwen3-VL answer
```

## Model roles

| Model | Role in this repository |
|---|---|
| `openai/clip-vit-base-patch32` | shared image/text embeddings, similarity, zero-shot classification |
| `Qwen/Qwen3-VL-2B-Instruct` | generative visual understanding and grounded answers |
| `Qwen/Qwen3-VL-Embedding-2B` | multimodal retrieval embeddings through SentenceTransformers |
| `Qwen/Qwen3-VL-Reranker-2B` | second-stage relevance scoring after retrieval |

> Current Qwen3-VL is natively integrated into Hugging Face Transformers. `trust_remote_code` is disabled by default and is **not required** for the normal Qwen3-VL path implemented here.

## 1. Installation

### Requirements

- Python **3.12**
- Git
- Enough disk space for the models you choose to download
- GPU strongly recommended for Qwen3-VL inference; CLIP can run on CPU for small experiments

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

Development / tests only:

```bash
python -m pip install -e ".[dev]"
```

CLIP experiments:

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

Full project:

```bash
python -m pip install -e ".[all,dev,notebooks]"
```

## 2. Qwen3-VL execution modes

### Option A - Hugging Face model ID / normal cache

The first run downloads the model if it is not already cached. Later runs reuse the Hugging Face cache.

Python:

```python
from vlm_engineering.qwen import QwenVLModel

model = QwenVLModel.from_hub("Qwen/Qwen3-VL-2B-Instruct")
answer = model.generate("data/example.jpg", "Describe the image precisely.")
print(answer)
```

CLI:

```bash
vlm-lab describe data/example.jpg \
  --prompt "Describe the image and list important visual details."
```

### Option B - explicit download, then offline/local inference

Download once:

```bash
vlm-lab download-model \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --output models/Qwen3-VL-2B-Instruct
```

Run from that directory only:

```bash
vlm-lab describe data/example.jpg \
  --model-path models/Qwen3-VL-2B-Instruct \
  --prompt "Describe the image precisely."
```

Python:

```python
from vlm_engineering.qwen import QwenVLModel

model = QwenVLModel.from_local("models/Qwen3-VL-2B-Instruct")
print(model.generate("data/example.jpg", "What is happening in this image?"))
```

`from_local()` sets `local_files_only=True`, so it does not silently fall back to the Hub.

### Option C - explicit remote custom code (advanced, normally unnecessary here)

```bash
vlm-lab describe data/example.jpg --trust-remote-code
```

Only enable this for a model you trust and that genuinely requires repository-provided Python code. It expands the security trust boundary.

## 3. Core Qwen3-VL use cases

### Image captioning / rich description

```bash
vlm-lab describe photo.jpg --prompt "Write a factual, detailed description."
```

### Visual question answering

```bash
vlm-lab describe dashboard.png --prompt "Which metric increased the most and what evidence supports it?"
```

### Diagram / architecture understanding

```bash
vlm-lab analyze architecture.png
```

The `analyze` command asks Qwen3-VL for stable JSON containing page type, title, summary, entities, relations, important text and uncertainties.

### Screenshot / UI understanding

Use the same `describe` or `analyze` interfaces for dashboards, application screenshots, forms and technical UI states.

### Tables and document pages

Render the page as an image and ask for structured extraction. For production documents, preserve native/OCR text separately and use the VLM for layout and relationships.

## 4. VLM + OCR/native parsing: hybrid, not replacement by default

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

## 5. Build a RAG-ready visual chunk

```bash
vlm-lab chunk data/page_001.png \
  --document-id architecture-v1 \
  --source-file architecture.pdf \
  --page 1 \
  --native-text-file data/page_001.txt
```

The output keeps traceability such as document ID, page, image reference, entities, relations and fused retrieval text.

## 6. Visual RAG when you only have a text embedder

This compatibility pattern is extremely useful:

```text
image/page -> Qwen3-VL description/JSON -> retrieval text -> text embedder -> vector DB
```

The VLM makes visual information searchable by converting it into a faithful text representation. See:

```bash
python examples/text_only_visual_rag.py
```

This does **not** mean the image should be discarded. Keep `image_ref` so the final VLM can inspect the original visual evidence after retrieval.

## 7. True multimodal embeddings with Qwen3-VL-Embedding

The retrieval model can embed text, image references, or mixed text+image inputs in a shared space.

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

SentenceTransformers integration is the intended path for this model family in this repository.

## 8. Reranking

Use the embedding model for broad recall, then rerank the top candidates:

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
  -> Qwen3-VL-Instruct  -> grounded answer with source/page
```

## 9. CLIP foundations

CLIP remains valuable because it teaches the shared embedding-space idea that modern multimodal retrieval systems build upon.

```python
from PIL import Image
from vlm_engineering.clip import CLIPEncoder

image = Image.open("data/example.jpg").convert("RGB")
clip = CLIPEncoder()
print(clip.zero_shot_classify(image, ["cat", "dog", "airplane"]))
```

## 10. Notebooks

Notebooks are **analysis clients**, not the implementation layer:

- `00_clip_foundations.ipynb`
- `01_qwen3_vl_quickstart.ipynb`
- `02_visual_document_understanding.ipynb`
- `03_multimodal_rag.ipynb`

All reusable logic lives under `src/vlm_engineering` and can be called without Jupyter.

## 11. Quality gates

```bash
python -m ruff check src tests examples
python -m mypy src/vlm_engineering
python -m pytest --cov=vlm_engineering --cov-report=term-missing
python -m pip_audit .
```

CI additionally validates notebook JSON, Docker build and CodeQL.

## 12. Docker

Build:

```bash
docker build -t vision-language-engineering-lab .
```

Mount your data and persistent model cache:

```bash
docker compose run --rm vlm-lab describe data/example.jpg
```

GPU execution depends on the host/container runtime. For serious Qwen inference, use an NVIDIA-enabled environment or a dedicated inference server such as vLLM.

## 13. Documentation

- `docs/pdf/Vision_Language_Engineering_EN.pdf`
- `docs/pdf/Ingenierie_Vision_Langage_FR.pdf`
- `docs/architecture.md`
- `docs/rag-patterns.md`
- `docs/model-loading.md`

## 14. Repository structure

```text
src/vlm_engineering/     production code
examples/                runnable examples
notebooks/               analysis/demonstration clients
tests/                   lightweight deterministic tests
docs/                    architecture, RAG guidance and bilingual course PDFs
legacy/                  previous educational notebooks, isolated from maintained code
models/                  local downloaded weights (gitignored)
data/                    user-provided images/pages (gitignored)
```

## License and third-party material

New maintained project code is distributed under Apache-2.0. Model weights are downloaded separately and keep their own licenses. Historical notebooks are isolated under `legacy/`; see `THIRD_PARTY_NOTICES.md` before redistributing them.
