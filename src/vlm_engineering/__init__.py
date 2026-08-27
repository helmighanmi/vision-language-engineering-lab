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
from .qwen.model import QwenVLModel
from .qwen.registry import (
    DEFAULT_QWEN_MODEL_SIZE,
    QWEN3_VL_INSTRUCT_MODELS,
    QwenModelPreset,
    resolve_qwen_model_id,
)
from .retrieval.rag import RAGAnswer, VisualRAGPipeline

__all__ = [
    "CLIPEncoder",
    "DEFAULT_QWEN_MODEL_SIZE",
    "NativePageContent",
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
    "fuse_native_and_visual",
    "resolve_qwen_model_id",
]
