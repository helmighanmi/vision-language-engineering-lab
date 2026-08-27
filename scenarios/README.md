<!--
Path: scenarios/README.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Runnable VLM Scenarios

These scripts are an application cookbook for `vision-language-engineering-lab`. They call the public package APIs directly; no notebook is required.

## Install

For all scenarios:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[all]"
```

Put your own images under `data/` (the directory is gitignored).

## Qwen model selection

Scenarios 01-05 and 09 accept the same generator choices:

```bash
# Default 2B
python scenarios/scenario_01_image_captioning.py data/example.jpg

# 4B preset
python scenarios/scenario_01_image_captioning.py data/example.jpg --model-size 4b

# 8B preset
python scenarios/scenario_01_image_captioning.py data/example.jpg --model-size 8b

# Any compatible Hugging Face model ID
python scenarios/scenario_01_image_captioning.py data/example.jpg \
  --model-id Qwen/Qwen3-VL-4B-Instruct

# Explicit local/offline directory
python scenarios/scenario_01_image_captioning.py data/example.jpg \
  --model-path models/Qwen3-VL-4B-Instruct
```

Start with **2B**. Move to **4B** or **8B** only after measuring quality on your own images and confirming you have enough memory/compute.

## Scenarios

| Script | What it demonstrates |
|---|---|
| `scenario_01_image_captioning.py` | rich image captioning / description |
| `scenario_02_visual_question_answering.py` | grounded visual question answering |
| `scenario_03_diagram_to_json.py` | diagram/screenshot -> validated structured JSON |
| `scenario_04_document_page_to_rag_chunk.py` | native text + VLM semantics -> traceable RAG chunk |
| `scenario_05_text_only_visual_rag.py` | VLM descriptions + ordinary text embedder -> visual RAG |
| `scenario_06_true_multimodal_retrieval.py` | direct text-to-image retrieval with Qwen3-VL-Embedding |
| `scenario_07_compare_qwen_presets.py` | compare 2B/4B/8B on the same prompt/image |
| `scenario_08_hub_vs_local_loading.py` | Hub/cache vs explicit offline/local loading |
| `scenario_09_batch_structured_analysis.py` | reuse one model for batch JSONL document/image analysis |

## Copy/paste examples

### 1. Image captioning

```bash
python scenarios/scenario_01_image_captioning.py data/example.jpg --model-size 2b
```

### 2. Visual question answering

```bash
python scenarios/scenario_02_visual_question_answering.py \
  data/architecture.png \
  "Which service writes to PostgreSQL?" \
  --model-size 4b
```

### 3. Diagram to structured JSON

```bash
python scenarios/scenario_03_diagram_to_json.py \
  data/architecture.png \
  --model-size 4b \
  --output output/architecture.json
```

### 4. Document page to RAG chunk

```bash
python scenarios/scenario_04_document_page_to_rag_chunk.py \
  data/page_001.png \
  --document-id architecture-v1 \
  --source-file architecture.pdf \
  --page 1 \
  --native-text-file data/page_001.txt \
  --parent-context "Backend architecture" \
  --model-size 4b \
  --output output/page_001.chunk.json
```

### 5. Visual RAG with only a text embedder

```bash
python scenarios/scenario_05_text_only_visual_rag.py \
  data/page_001.png data/page_002.png data/page_003.png \
  --question "Which component uses Redis?" \
  --model-size 2b
```

This pattern is useful when your existing vector store already uses a text embedding model: Qwen first turns visual content into retrieval-quality text/structure, and the normal text embedder indexes it.

### 6. True multimodal retrieval

```bash
python scenarios/scenario_06_true_multimodal_retrieval.py \
  data/architecture.png data/dashboard.png data/table.png \
  --query "Find the diagram that contains a cache and a database" \
  --top-k 2
```

This scenario uses `Qwen/Qwen3-VL-Embedding-2B`. It is a retrieval model, not the 2B/4B/8B Instruct generator.

### 7. Compare Qwen3-VL presets

```bash
python scenarios/scenario_07_compare_qwen_presets.py \
  data/architecture.png \
  --sizes 2b 4b 8b
```

**Warning:** this can download several large checkpoints and needs substantially more memory than a single 2B run. You can compare only two sizes, e.g. `--sizes 2b 4b`.

### 8. Hub/cache vs explicit local model

```bash
# Hub/cache
python scenarios/scenario_08_hub_vs_local_loading.py \
  data/example.jpg --mode hub --model-size 4b

# Download/use an explicit directory
python scenarios/scenario_08_hub_vs_local_loading.py \
  data/example.jpg --mode local --model-size 4b \
  --model-path models/Qwen3-VL-4B-Instruct --download-if-missing
```

After downloading, set `HF_HUB_OFFLINE=1` if you want to enforce an offline environment.

### 9. Batch structured analysis

```bash
python scenarios/scenario_09_batch_structured_analysis.py \
  data/page_001.png data/page_002.png data/page_003.png \
  --model-size 4b \
  --output output/visual_analysis.jsonl
```

The model is loaded once and reused across the batch.

## Choosing a scenario

- Need a **caption or image explanation** -> scenario 01.
- Need to **ask a question about one image** -> scenario 02.
- Need **machine-validatable diagram structure** -> scenario 03.
- Building **document RAG with native/OCR text** -> scenario 04.
- Your current RAG has **text embeddings only** -> scenario 05.
- You need **direct cross-modal retrieval** -> scenario 06.
- You want to measure **2B vs 4B vs 8B** -> scenario 07.
- You need **offline/reproducible model deployment** -> scenario 08.
- You need **many pages/images processed with one model load** -> scenario 09.

## Production notes

- Do not treat larger parameter count as automatic quality. Evaluate on your own dataset.
- Preserve native/OCR text when exact strings, numbers, IDs, and identifiers matter; use the VLM to add layout and relationships.
- Keep the original image reference in RAG metadata so the final VLM can re-check visual evidence.
- Avoid enabling `trust_remote_code` unless the model requires it and you have reviewed the repository.
- For high-throughput serving, prefer a dedicated inference server rather than loading one large model per request/process.
