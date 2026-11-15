# Tests for documents extended
from collections.abc import Sequence
from pathlib import Path

from vlm_engineering.documents.visual_analysis import analyze_visual_document


class StaticGenerator:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def generate(
        self, images: Sequence[str | Path], prompt: str, *, max_new_tokens: int = 512
    ) -> str:
        self.calls.append([str(image) for image in images])
        return """{"page_type":"diagram","title":"T","summary":"A calls B","entities":["A","B"],"relations":[{"source":"A","relation":"calls","target":"B","evidence":"arrow"}],"important_text":[],"uncertainties":[]}"""


def test_analyze_with_static_generator() -> None:
    generator = StaticGenerator()
    result = analyze_visual_document(generator, "page.png")
    assert result.page_type == "diagram"
