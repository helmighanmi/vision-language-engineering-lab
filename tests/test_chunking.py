# Path: tests/test_chunking.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from vlm_engineering.documents import NativePageContent, TextChunker, VisualAnalysis, VisualChunkBuilder


def test_text_chunker_rejects_non_progressing_overlap() -> None:
    try:
        TextChunker(max_chars=100, overlap_chars=100)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


def test_visual_chunk_is_traceable() -> None:
    native = NativePageContent("doc", 2, "demo.pdf", native_text="Exact ID: ABC-42", image_ref="page.png")
    visual = VisualAnalysis("diagram", "Architecture", "Service A calls DB", entities=("Service A", "DB"))
    chunk = VisualChunkBuilder().build(native, visual, parent_context="Backend")
    assert chunk.page == 2
    assert chunk.image_ref == "page.png"
    assert "ABC-42" in chunk.text
    assert "Service A calls DB" in chunk.text
