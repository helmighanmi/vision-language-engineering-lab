# Path: scenarios/scenario_01_image_captioning.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Scenario 01: create a rich caption/description for one image."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_qwen_model_args, build_qwen, model_label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a detailed image caption with Qwen3-VL.")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--prompt",
        default=(
            "Describe this image precisely. Mention the main objects, visible text, layout, "
            "relationships, and any uncertainty."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    add_qwen_model_args(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = build_qwen(args)
    print(f"Model: {model_label(model)}")
    print(model.generate(args.image, args.prompt, max_new_tokens=args.max_new_tokens))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
