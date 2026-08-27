<!--
Path: docs/decisions/002-notebooks-are-clients.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# ADR-002: Notebooks are clients, not production code

## Decision

All reusable CLIP, Qwen, document analysis, chunking and RAG logic lives under `src/vlm_engineering`.

## Consequence

The same implementation can be called from Python, CLI, notebooks, services or batch jobs. CI does not need Jupyter to test the production core.
