<!--
Path: CHANGELOG.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Changelog

## Unreleased - Configurable Qwen3-VL model selection

- Added first-class `2b`, `4b`, and `8b` Qwen3-VL Instruct presets.
- Kept `Qwen/Qwen3-VL-2B-Instruct` as the default model.
- Added arbitrary compatible Hugging Face `--model-id` support and explicit local/offline `--model-path`.
- Added `vlm-lab models` plus preset-aware model downloads.
- Added model-selection tests and user guidance for sizing, memory, offline loading and troubleshooting.
- Added a nine-scenario application cookbook covering captioning, VQA, diagram JSON, RAG chunking, text-only visual RAG, multimodal retrieval, model comparison, offline loading and batch analysis.

## 1.0.0 - Production-oriented refactor

- Repositioned the repository as Vision-Language Engineering Lab.
- Added reusable CLIP encoder APIs.
- Added Qwen3-VL-2B-Instruct Hub/cache and explicit local/offline loading.
- Added explicit model downloader and secure `trust_remote_code` default.
- Added structured visual-document analysis, native+visual fusion and RAG-ready chunks.
- Added text-only visual RAG plus Qwen multimodal embedding/reranking adapters.
- Added CLI, examples, tests, Docker, CI/security workflows and bilingual teaching PDFs.
