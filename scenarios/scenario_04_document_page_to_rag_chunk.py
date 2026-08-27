# Path: scenarios/scenario_04_document_page_to_rag_chunk.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Scenario 04: fuse exact native text with VLM semantics into one traceable RAG chunk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import add_qwen_model_args, build_qwen, model_label
from vlm_engineering.documents import NativePageContent, VisualChunkBuilder, analyze_visual_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a RAG-ready chunk from a rendered document page.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--page", type=int, required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--native-text-file", type=Path)
    parser.add_argument("--parent-context", default="")
    parser.add_argument("--output", type=Path)
    add_qwen_model_args(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = build_qwen(args)
    analysis = analyze_visual_document(model, str(args.image))
    native_text = (
        args.native_text_file.read_text(encoding="utf-8") if args.native_text_file else ""
    )
    native = NativePageContent(
        document_id=args.document_id,
        page=args.page,
        source_file=args.source_file,
        native_text=native_text,
        title=args.title,
        image_ref=str(args.image),
    )
    chunk = VisualChunkBuilder().build(native, analysis, parent_context=args.parent_context)
    payload = json.dumps(chunk.to_dict(), indent=2, ensure_ascii=False)
    print(f"Model: {model_label(model)}")
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
