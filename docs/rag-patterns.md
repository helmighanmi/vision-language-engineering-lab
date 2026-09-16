<!--
Path: docs/rag-patterns.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Visual RAG patterns

## Pattern A - VLM description + text embedder

Use Qwen3-VL to turn an image into a rich factual description, then embed that description with any text embedding model. This is the best compatibility path when the existing RAG stack is text-only.

## Pattern B - native text + VLM structure

Keep deterministic/native extraction for exact strings, IDs and numbers. Use the VLM to recover layout, arrows, groups, tables and visual relations. Fuse both representations before indexing.

## Pattern C - true multimodal retrieval

Use QwenVLEmbedder to embed text, local images or text+image inputs. MultimodalRAGPipeline retrieves candidate_k, optionally scores candidates with QwenVLReranker, selects top_k, and attaches all selected original images to generation. Video is roadmap-only. See [implemented API and migration](../README.md#13-multimodal-retrieval-and-rag--implemented-in-v030).

## OCR guidance

Do not assume a VLM always replaces OCR. OCR/native parsing is often cheaper and more exact for long literal text. A VLM adds visual semantics that OCR loses. Production document systems usually benefit from a hybrid pipeline.
