<!--
Path: docs/qwen-model-selection.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Qwen3-VL model selection

## Default

The package defaults to:

```text
Qwen/Qwen3-VL-2B-Instruct
```

This is intentional: the default should be usable for learning and prototyping without forcing every user to start from the largest model.

## Presets

| Alias | Model | Parameter class | Use it when... |
|---|---|---:|---|
| `2b` | `Qwen/Qwen3-VL-2B-Instruct` | ~2B | you are developing, learning, or resource constrained |
| `4b` | `Qwen/Qwen3-VL-4B-Instruct` | ~4B | you want a quality/resource middle ground |
| `8b` | `Qwen/Qwen3-VL-8B-Instruct` | ~8B | you can afford higher memory/latency for stronger capacity |

## Do not choose by parameter count alone

Evaluate on your own tasks:

- diagram relationship extraction
- OCR/text accuracy
- JSON validity
- visual question answering
- RAG retrieval quality
- final grounded answer quality
- latency and memory

For RAG, a smaller VLM with better chunking/retrieval can outperform a larger model fed poor evidence.

## Rough memory intuition

For BF16/FP16 weights alone, 2 bytes per parameter gives a useful lower-bound intuition:

```text
2B -> ~4 GB weights only
4B -> ~8 GB weights only
8B -> ~16 GB weights only
```

This is **not** total VRAM. Add vision-model weights, KV cache, activations, image tokens, framework overhead and batching.

## Extensibility

The aliases are conveniences. Advanced users can always pass a compatible Hugging Face model ID:

```bash
vlm-lab describe image.jpg --model-id Qwen/Qwen3-VL-4B-Instruct
```

This avoids coupling package releases to every future compatible Qwen checkpoint.
