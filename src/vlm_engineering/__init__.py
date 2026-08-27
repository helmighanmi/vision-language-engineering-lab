# Path: src/vlm_engineering/__init__.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Production-oriented Vision-Language engineering toolkit."""

from .documents.chunking import TextChunker, VisualChunkBuilder
from .documents.fusion import fuse_native_and_visual
from .documents.schemas import NativePageContent, VisualAnalysis, VisualChunk, VisualRelation
from .qwen.model import QwenVLModel

__all__ = [
    "NativePageContent",
    "QwenVLModel",
    "TextChunker",
    "VisualAnalysis",
    "VisualChunk",
    "VisualChunkBuilder",
    "VisualRelation",
    "fuse_native_and_visual",
]
