# Path: tests/test_visual_analysis.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

from collections.abc import Sequence
from pathlib import Path


class FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def generate(
        self, images: Sequence[str | Path], prompt: str, *, max_new_tokens: int = 512
    ) -> str:
        self.calls.append([str(image) for image in images])
        return """{"page_type":"diagram","title":"T","summary":"A calls B","entities":["A","B"],"relations":[{"source":"A","relation":"calls","target":"B","evidence":"arrow"}],"important_text":[],"uncertainties":}"""
