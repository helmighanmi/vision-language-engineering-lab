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
