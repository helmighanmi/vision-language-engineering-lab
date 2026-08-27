# Path: scenarios/scenario_09_batch_structured_analysis.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Scenario 09: analyze many images/pages with one loaded Qwen model and write JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import add_qwen_model_args, build_qwen, model_label

from vlm_engineering.documents import analyze_visual_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch structured VLM analysis to JSONL.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/visual_analysis.jsonl"))
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first failed image instead of recording the error and continuing.",
    )
    add_qwen_model_args(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = build_qwen(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as handle:
        for index, image in enumerate(args.images, start=1):
            try:
                analysis = analyze_visual_document(model, str(image))
                record = {
                    "image": str(image),
                    "model": model.model_source,
                    "analysis": analysis.to_dict(),
                }
                print(f"OK {index}/{len(args.images)}: {image}")
            except Exception as exc:  # keep long batches useful while preserving the failure
                if args.fail_fast:
                    raise
                record = {
                    "image": str(image),
                    "model": model.model_source,
                    "error": str(exc),
                }
                print(f"ERROR {index}/{len(args.images)}: {image}: {exc}")
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Model: {model_label(model)}")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
