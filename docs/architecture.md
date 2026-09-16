<!--
Path: docs/architecture.md
Author: GHANMI Helmi
Current Role: AI Engineer
Past Role: Researcher in Applied Mathematics
Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi
-->

# Architecture

The maintained architecture separates **model adapters**, **document semantics**, **retrieval**, and **interfaces**.

```text
Image / rendered document page
        |
        +--> CLIP -> shared embeddings -> similarity / zero-shot retrieval
        |
        +--> Qwen3-VL-Instruct -> description / VQA / structured JSON
                                  |
Native text ----------------------+--> fusion -> visual chunks
                                                  |
                      +---------------------------+--------------------+
                      |                                                |
              text embedder                                Qwen3-VL-Embedding
                      |                                                |
                      +---------------> retrieval <--------------------+
                                            |
                                   optional reranker
                                            |
                                      Qwen3-VL final
                                            |
                                      grounded answer
```

The notebook layer imports these APIs and contains no production model-loading logic.

## v0.3 implementation

The core imports only NumPy and Pillow. `inputs` owns the text/image contract,
`operational` maps recognized runtime failures, and retrieval adapters share typed
loading options and a lazy backend loader. `VisualRAGPipeline` keeps its text
retrieval role; `MultimodalRAGPipeline` indexes text plus original images, optionally
reranks candidates and calls `generate_images` with all selected evidence. Both
use the replacing in-memory cosine index. Runtime YAML is optional and validated
without importing ML backends; the existing argparse CLI routes the new commands.
