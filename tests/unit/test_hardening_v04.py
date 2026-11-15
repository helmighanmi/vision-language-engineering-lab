# Regression tests for v0.4 reliability hardening.
"""Ensure the code changes from the code review don't break existing functionality."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from PIL import Image

from vlm_engineering import inputs
from vlm_engineering.documents.chunking import VisualChunkBuilder
from vlm_engineering.documents.schemas import (
    NativePageContent,
    VisualChunk,
    VisualAnalysis,
)
from vlm_engineering.documents.visual_analysis import analyze_visual_document
from vlm_engineering.retrieval import TextEmbedder
from vlm_engineering.service.concurrency import ServiceBusyError
from vlm_engineering.service.runtime import ServiceRuntime
from vlm_engineering.service.config import ServiceConfig


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
        "prefix\n```json\n{\"a\": 1}\n```\ntrailing prose",
        'first {"a": 1} second {"b": 2}',
        '{"outer": {"inner": "closing } inside a string"}}',
    ],
)
def test_hardening(raw: str) -> None:
    """"""
    config = ServiceConfig(media_root=Path("/tmp/media"))
    runtime = ServiceRuntime(config)
    generator = SequenceGenerator('{"page_type":"diagram"}')
    result = analyze_visual_document(generator, "page.png")
    assert result.page_type == "diagram"
