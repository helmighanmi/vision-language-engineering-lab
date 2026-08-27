# Path: scenarios/common.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Shared CLI helpers for runnable scenario scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_engineering.qwen import DEFAULT_QWEN_MODEL_SIZE, QWEN3_VL_INSTRUCT_MODELS, QwenVLModel


def add_qwen_model_args(parser: argparse.ArgumentParser) -> None:
    """Add the same model-selection contract used by the package CLI."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model-size",
        choices=tuple(QWEN3_VL_INSTRUCT_MODELS),
        default=None,
        help="Qwen3-VL preset: 2b (default), 4b, or 8b.",
    )
    group.add_argument(
        "--model-id",
        help="Explicit compatible Hugging Face model ID.",
    )
    group.add_argument(
        "--model-path",
        type=Path,
        help="Explicit local model directory for offline/local-files-only loading.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map value. Default: auto.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Transformers dtype value. Default: auto.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Opt in to custom repository code. Current Qwen3-VL does not require it.",
    )


def build_qwen(args: argparse.Namespace) -> QwenVLModel:
    """Create a Qwen model from preset, explicit Hub ID, or local path."""
    common = {
        "device_map": args.device_map,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.model_path is not None:
        return QwenVLModel.from_local(args.model_path, **common)
    if args.model_id:
        return QwenVLModel.from_hub(args.model_id, **common)
    return QwenVLModel.from_preset(args.model_size or DEFAULT_QWEN_MODEL_SIZE, **common)


def model_label(model: QwenVLModel) -> str:
    """Return a human-readable selected model source."""
    mode = "local/offline" if model.local_files_only else "Hub/cache"
    return f"{model.model_source} [{mode}]"
