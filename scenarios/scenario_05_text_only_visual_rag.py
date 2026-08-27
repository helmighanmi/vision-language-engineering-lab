# Path: scenarios/scenario_05_text_only_visual_rag.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Scenario 05: make images searchable when your existing RAG stack only embeds text."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import add_qwen_model_args, build_qwen, model_label

from vlm_engineering.documents import NativePageContent, VisualChunkBuilder, analyze_visual_document
from vlm_engineering.retrieval import TextEmbedder, VisualRAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Describe images with Qwen3-VL, embed the derived text, then answer with visual RAG."
    )
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--text-embedder",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    add_qwen_model_args(parser)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = build_qwen(args)
    chunks = []
    for index, image in enumerate(args.images, start=1):
        analysis = analyze_visual_document(model, str(image))
        native = NativePageContent(
            document_id=image.stem,
            page=1,
            source_file=image.name,
            image_ref=str(image),
        )
        chunks.append(
            VisualChunkBuilder().build(
                native,
                analysis,
                parent_context="Visual evidence indexed through VLM-derived text.",
            )
        )
        print(f"Prepared {index}/{len(args.images)}: {image}")

    pipeline = VisualRAGPipeline(TextEmbedder(args.text_embedder), model)
    pipeline.index_chunks(chunks)
    result = pipeline.answer(args.question, top_k=args.top_k)
    print(f"Model: {model_label(model)}")
    print("\nANSWER\n------")
    print(result.answer)
    print("\nRETRIEVED EVIDENCE\n------------------")
    for chunk in result.chunks:
        print(f"- {chunk.source_file} (page {chunk.page}) :: {chunk.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
