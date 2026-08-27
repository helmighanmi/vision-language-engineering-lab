# Path: scenarios/scenario_07_compare_qwen_presets.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Scenario 07: compare Qwen3-VL 2B, 4B and 8B on the same image/prompt."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

from vlm_engineering.qwen import QWEN3_VL_INSTRUCT_MODELS, QwenVLModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare selected Qwen3-VL Instruct presets.")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--sizes",
        nargs="+",
        choices=tuple(QWEN3_VL_INSTRUCT_MODELS),
        default=["2b", "4b", "8b"],
        help="Preset sizes to compare. Default: 2b 4b 8b.",
    )
    parser.add_argument("--prompt", default="Describe this image precisely and identify important relationships.")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="auto")
    return parser


def release_accelerator_memory() -> None:
    """Best-effort cleanup between large model comparisons."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    args = build_parser().parse_args()
    print("WARNING: comparing multiple models can require substantial download time and GPU memory.\n")
    for size in args.sizes:
        preset = QWEN3_VL_INSTRUCT_MODELS[size]
        print(f"=== {size.upper()} :: {preset.model_id} ({preset.parameter_class}) ===")
        started = time.perf_counter()
        try:
            model = QwenVLModel.from_preset(
                size,
                device_map=args.device_map,
                dtype=args.dtype,
            )
            answer = model.generate(
                args.image,
                args.prompt,
                max_new_tokens=args.max_new_tokens,
            )
            elapsed = time.perf_counter() - started
            print(answer)
            print(f"Elapsed: {elapsed:.2f}s\n")
            del model
        except Exception as exc:  # hardware/model-download errors are scenario-specific
            elapsed = time.perf_counter() - started
            print(f"FAILED after {elapsed:.2f}s: {exc}\n")
        finally:
            release_accelerator_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
