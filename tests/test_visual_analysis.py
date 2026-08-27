# Path: tests/test_visual_analysis.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from vlm_engineering.documents.visual_analysis import analyze_visual_document


class FakeGenerator:
    def generate(self, image: str, prompt: str, **kwargs: object) -> str:
        return '''{"page_type":"diagram","title":"T","summary":"A calls B","entities":["A","B"],"relations":[{"source":"A","relation":"calls","target":"B","evidence":"arrow"}],"important_text":[],"uncertainties":[]}'''


def test_structured_analysis() -> None:
    result = analyze_visual_document(FakeGenerator(), "page.png")
    assert result.page_type == "diagram"
    assert result.relations[0].target == "B"
