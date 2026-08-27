# Path: src/vlm_engineering/documents/__init__.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Document-oriented VLM analysis and semantic visual chunking."""

from .chunking import TextChunker, VisualChunkBuilder
from .fusion import fuse_native_and_visual
from .schemas import NativePageContent, VisualAnalysis, VisualChunk, VisualRelation
from .visual_analysis import analyze_visual_document

__all__ = [
    "NativePageContent",
    "TextChunker",
    "VisualAnalysis",
    "VisualChunk",
    "VisualChunkBuilder",
    "VisualRelation",
    "analyze_visual_document",
    "fuse_native_and_visual",
]
