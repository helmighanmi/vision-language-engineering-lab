# Path: scenarios/scenario_08_hub_vs_local_loading.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Scenario 08: run the same Qwen model from Hub/cache or an explicit local directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_engineering.qwen import (
    QWEN3_VL_INSTRUCT_MODELS,
    QwenVLModel,
    default_model_directory,
    download_model_snapshot,
    resolve_qwen_model_id,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Hub/cache and explicit local Qwen loading.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--mode", choices=("hub", "local"), required=True)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model-size", choices=tuple(QWEN3_VL_INSTRUCT_MODELS), default=None)
    model_group.add_argument("--model-id")
    parser.add_argument("--model-path", type=Path, help="Local directory; defaults to models/<name>.")
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--prompt", default="Describe this image precisely.")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_id = resolve_qwen_model_id(model_size=args.model_size, model_id=args.model_id)

    if args.mode == "hub":
        model = QwenVLModel.from_hub(model_id, device_map=args.device_map, dtype=args.dtype)
    else:
        model_path = args.model_path or default_model_directory(model_id)
        if not model_path.exists() and args.download_if_missing:
            download_model_snapshot(model_id, output_dir=model_path, revision=args.revision)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Local model not found at {model_path}. Use --download-if-missing or run "
                f"'vlm-lab download-model --model-id {model_id}'."
            )
        model = QwenVLModel.from_local(model_path, device_map=args.device_map, dtype=args.dtype)

    print(f"Model source: {model.model_source}")
    print(f"Local files only: {model.local_files_only}")
    print(model.generate(args.image, args.prompt, max_new_tokens=args.max_new_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
