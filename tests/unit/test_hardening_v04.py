# Regression tests for v0.4 reliability hardening.
"""Ensure the code changes from the code review don't break existing functionality."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from vlm_engineering.documents.visual_analysis import analyze_visual_document


class SequenceGenerator:
    def __init__(self, *responses: object) -> None:
        self.responses = list(responses)
        self.calls: list[list[str]] = []

    def generate(
        self, images: Sequence[str | Path], prompt: str, *, max_new_tokens: int = 512
    ) -> str:
        self.calls.append([str(image) for image in images])
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return str(value)


@pytest.mark.parametrize(
    "raw",
    [
        '{"a": 1}',
        "prefix\\n```json\\n{\"a\": 1}\\n```\\ntrailing prose",
        'first {"a": 1} second {"b": 2}',
        '{"outer": {"inner": "closing } inside a string"}}',
    ],
)
def test_hardening(raw: str) -> None:
    generator = SequenceGenerator('{"page_type":"diagram"}')
    result = analyze_visual_document(generator, "page.png")
    assert result.page_type == "diagram"
