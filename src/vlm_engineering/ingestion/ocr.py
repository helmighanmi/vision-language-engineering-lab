# Path: src/vlm_engineering/ingestion/ocr.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Pluggable OCR interface. Implementations convert an image path to text."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class OCRAdapter(Protocol):
    """Recognize text in an image file."""

    def recognize(self, image_path: str | Path) -> str:
        """Return the OCR text for ``image_path``."""
        ...


class NullOCR:
    """No-op OCR that returns an empty string."""

    def recognize(self, image_path: str | Path) -> str:
        return ""
