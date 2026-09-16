# Path: src/vlm_engineering/inputs.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Explicit text/image inputs and eager local-image validation (no model imports)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypeAlias
from urllib.parse import unquote, urlparse

from PIL import Image, UnidentifiedImageError

from .exceptions import InputValidationError

MultimodalInput: TypeAlias = str | Mapping[str, str]
IMAGE_FORMATS = frozenset({"PNG", "JPEG", "WEBP", "BMP", "TIFF"})


def positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise InputValidationError(f"{name} must be a positive integer.")


def boolean(value: bool, name: str) -> None:
    if type(value) is not bool:
        raise InputValidationError(f"{name} must be a boolean.")


def nonempty_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InputValidationError(f"{name} must be a non-empty string.")


def validate_image(value: str | Path) -> str:
    """Accept local paths/file URIs, decode before model loading, return a normal path.

    Remote image downloads and animated/multi-frame images are deliberately outside
    the retrieval contract. Materialize them locally before calling this API.
    """
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise InputValidationError("image must be a non-empty local path.")
    raw = str(value)
    if raw.startswith("file://"):
        uri = urlparse(raw)
        if uri.netloc not in ("", "localhost") or uri.query or uri.fragment:
            raise InputValidationError("image file URI must refer to a local file without query or fragment.")
        raw = unquote(uri.path)
    elif "://" in raw or raw.startswith("data:"):
        raise InputValidationError("image must be a local file; download remote images first.")
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if not path.is_file():
        raise InputValidationError(f"Image path is not a file: {path}")
    try:
        with Image.open(path) as image:
            if image.format not in IMAGE_FORMATS or getattr(image, "n_frames", 1) != 1:
                raise InputValidationError(
                    "Unsupported image format or multiple frames; use a single PNG/JPEG/WEBP/BMP/TIFF."
                )
            image.verify()
        with Image.open(path) as image:
            image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise InputValidationError(
            f"Unsupported or corrupt image: {path}. Re-export as PNG or JPEG."
        ) from exc
    return str(path)


def normalize_input(value: MultimodalInput) -> dict[str, str]:
    """Strings mean text; ambiguous path-like strings require explicit image/text keys."""
    if isinstance(value, str):
        nonempty_string(value, "text")
        raw = value.strip()
        if (
            raw.startswith(("/", "./", "../", "~", "data:"))
            or "://" in raw
            or (
                Path(raw).suffix.lower()
                in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif", ".svg"}
            )
        ):
            raise InputValidationError('Ambiguous string input; use {"image": path} or {"text": text}.')
        value = {"text": value}
    if not isinstance(value, Mapping) or not value or set(value) - {"text", "image"}:
        raise InputValidationError("Input must contain only text and/or image fields, and cannot be empty.")
    result: dict[str, str] = {}
    for key, item in value.items():
        nonempty_string(item, key)
        result[key] = validate_image(item) if key == "image" else item
    return result


def normalize_inputs(values: Sequence[MultimodalInput]) -> list[dict[str, str]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise InputValidationError("inputs must be a sequence of text/image inputs.")
    return [normalize_input(value) for value in values]
