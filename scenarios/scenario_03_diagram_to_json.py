# Path: scenarios/scenario_03_diagram_to_json.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Scenario 03: convert a technical diagram into validated structured JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import add_qwen_model_args, build_qwen, model_label

from vlm_engineering.documents import analyze_visual_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract RAG-ready JSON from a diagram or screenshot.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, help="Optional JSON output file.")
    add_qwen_model_args(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = build_qwen(args)
    analysis = analyze_visual_document(model, str(args.image))
    payload = json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False)
    print(f"Model: {model_label(model)}")
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
