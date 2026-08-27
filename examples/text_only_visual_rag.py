# Path: examples/text_only_visual_rag.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Index VLM descriptions with a normal text embedder when no multimodal embedder is available."""

from vlm_engineering.documents import NativePageContent, VisualAnalysis, VisualChunkBuilder
from vlm_engineering.qwen import QwenVLModel
from vlm_engineering.retrieval import TextEmbedder, VisualRAGPipeline

chunks = [
    VisualChunkBuilder().build(
        NativePageContent("demo", 1, "demo.pdf", image_ref="data/page_001.png"),
        VisualAnalysis(
            page_type="diagram",
            title="Payment architecture",
            summary="API Gateway calls Payment Service, which uses Redis and PostgreSQL.",
            entities=("API Gateway", "Payment Service", "Redis", "PostgreSQL"),
        ),
    )
]
pipeline = VisualRAGPipeline(TextEmbedder(), QwenVLModel.from_hub())
pipeline.index_chunks(chunks)
print(pipeline.answer("Which component uses PostgreSQL?").answer)
