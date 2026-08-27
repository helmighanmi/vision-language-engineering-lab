# Path: scenarios/scenario_02_visual_question_answering.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Scenario 02: ask a grounded question about an image."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_qwen_model_args, build_qwen, model_label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Visual question answering with Qwen3-VL.")
    parser.add_argument("image", type=Path)
    parser.add_argument("question", help="Question that must be answered from the image.")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    add_qwen_model_args(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = build_qwen(args)
    prompt = (
        "Answer the question using only evidence visible in the image. "
        "If the answer is not visible or is ambiguous, say so.\n\n"
        f"QUESTION: {args.question}"
    )
    print(f"Model: {model_label(model)}")
    print(model.generate(args.image, prompt, max_new_tokens=args.max_new_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
