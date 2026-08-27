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
    """Extract the first JSON object from a model response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```")
        stripped = stripped.rsplit("```", 1)[0].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("No JSON object found in model output.")
    parsed = json.loads(stripped[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed
