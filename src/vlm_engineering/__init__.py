# Path: src/vlm_engineering/__init__.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Production-oriented Vision-Language engineering toolkit."""

from .clip.encoder import CLIPEncoder, ZeroShotPrediction
from .documents.chunking import TextChunker, VisualChunkBuilder
from .documents.fusion import fuse_native_and_visual
from .documents.schemas import (
    NativePageContent,
    VisualAnalysis,
    VisualChunk,
    VisualRelation,
)
from .extraction import (
    NUEXTRACT3_MODEL,
    NuExtract3Extractor,
    NuExtract3ModelPreset,
    create_nuextract3_extractor,
    load_extraction_template,
)
from .qwen.model import QwenVLModel
from .qwen.registry import (
    DEFAULT_QWEN_MODEL_SIZE,
    QWEN3_VL_INSTRUCT_MODELS,
    QwenModelPreset,
    resolve_qwen_model_id,
)
from .retrieval import MultimodalRAGPipeline, QwenVLEmbedder, QwenVLReranker
from .retrieval.rag import RAGAnswer, VisualRAGPipeline

__all__ = [
    "QwenVLEmbedder",
    "QwenVLReranker",
    "MultimodalRAGPipeline",
    "CLIPEncoder",
    "DEFAULT_QWEN_MODEL_SIZE",
    "NativePageContent",
    "NUEXTRACT3_MODEL",
    "NuExtract3Extractor",
    "NuExtract3ModelPreset",
    "QWEN3_VL_INSTRUCT_MODELS",
    "QwenModelPreset",
    "QwenVLModel",
    "RAGAnswer",
    "TextChunker",
    "VisualAnalysis",
    "VisualChunk",
    "VisualChunkBuilder",
    "VisualRAGPipeline",
    "VisualRelation",
    "ZeroShotPrediction",
    "create_nuextract3_extractor",
    "fuse_native_and_visual",
    "load_extraction_template",
    "resolve_qwen_model_id",
]
