# Path: src/vlm_engineering/utils.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Shared utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def to_image_reference(value: str | Path) -> str:
    """Convert a local path to a file URI while keeping HTTP(S) URLs unchanged."""
    raw = str(value)
    if raw.startswith(("http://", "https://", "file://")):
        return raw
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path.as_uri()


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object from a model response.

    Model responses often wrap JSON in Markdown fences, add prose before or
    after it, or emit several objects. Scanning for a parseable object with
    ``JSONDecoder.raw_decode`` handles all of those cases without relying on a
    brittle first-brace/last-brace slice.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("No JSON object found in model output.")

    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("No JSON object found in model output.")
