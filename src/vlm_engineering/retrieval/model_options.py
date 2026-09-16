# Path: src/vlm_engineering/retrieval/model_options.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Shared typed loading options; static Qwen retrieval presets remain Python data."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ..exceptions import InputValidationError, ModelLoadError
from ..inputs import boolean, nonempty_string

RetrievalKind = Literal["Embedding", "Reranker"]
EMBEDDING_DIMENSIONS = {"Qwen/Qwen3-VL-Embedding-2B": 2048, "Qwen/Qwen3-VL-Embedding-8B": 4096}


@dataclass(frozen=True)
class RetrievalModelOptions:
    model_size: str | None = None
    model_id: str | None = None
    model_path: str | None = None
    revision: str | None = None
    device: str | None = None
    cache_folder: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False

    def __post_init__(self) -> None:
        for name in ("model_size", "model_id", "model_path", "revision", "device", "cache_folder"):
            value = getattr(self, name)
            if value is not None:
                nonempty_string(value, name)
        for name in ("trust_remote_code", "local_files_only"):
            boolean(getattr(self, name), name)
        if sum(v is not None for v in (self.model_size, self.model_id, self.model_path)) > 1:
            raise InputValidationError("model_size/model_id/model_path are mutually exclusive.")
        if self.model_size is not None and self.model_size.lower() not in ("2b", "8b"):
            raise InputValidationError("model_size must be 2b or 8b for retrieval models.")

    def source(self, kind: RetrievalKind) -> str:
        if self.model_path is not None:
            return str(Path(self.model_path).expanduser().resolve())
        return self.model_id or f"Qwen/Qwen3-VL-{kind}-{(self.model_size or '2b').upper()}"

    def backend_kwargs(self) -> dict[str, Any]:
        values = asdict(self)
        for name in ("model_size", "model_id", "model_path"):
            values.pop(name)
        values["local_files_only"] = self.local_files_only or self.model_path is not None
        return values


def validate_snapshot(source: str) -> None:
    """Structural preflight for supported root-layout HF snapshots, not a checksum audit."""
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Local model directory not found: {path}")
    if not path.is_dir():
        raise InputValidationError(f"Model path is not a directory: {path}")
    required = ["config.json", "tokenizer_config.json", "preprocessor_config.json"]
    missing = [name for name in required if not (path / name).is_file()]
    if not (path / "tokenizer.json").is_file():
        missing.append("tokenizer.json")
    weights = list(path.glob("*.safetensors"))
    if not weights:
        missing.append("*.safetensors")
    try:
        for name in required:
            if (path / name).is_file():
                if not isinstance(json.loads((path / name).read_text()), dict):
                    missing.append(f"valid object in {name}")
        index = path / "model.safetensors.index.json"
        if index.exists():
            data = json.loads(index.read_text())
            weight_map = data.get("weight_map") if isinstance(data, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                missing.append("valid weight_map")
            else:
                for name in weight_map.values():
                    if not isinstance(name, str) or Path(name).name != name or not (path / name).is_file():
                        missing.append(f"weight shard {name}")
    except (ValueError, UnicodeError) as exc:
        raise ModelLoadError(
            f"Invalid local snapshot metadata in {path}; download the complete snapshot again."
        ) from exc
    if any(file.stat().st_size == 0 for file in weights):
        missing.append("non-empty weights")
    if missing:
        raise ModelLoadError(
            f"Incomplete local model snapshot {path}: missing {', '.join(missing)}. Download the complete snapshot again."
        )


def validate_dimensions(dimensions: int | None, maximum: int | None) -> None:
    if dimensions is not None and (
        type(dimensions) is not int or dimensions < 64 or dimensions > (maximum or 4096)
    ):
        raise InputValidationError(f"dimensions must be an integer between 64 and {maximum or 4096}.")
