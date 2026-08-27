# Path: scenarios/scenario_06_true_multimodal_retrieval.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Scenario 06: retrieve images directly with Qwen3-VL multimodal embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

from vlm_engineering.retrieval import InMemoryVectorIndex, QwenMultimodalEmbedder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Text-to-image retrieval with multimodal embeddings.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-VL-Embedding-2B",
        help="Multimodal embedding model ID.",
    )
    parser.add_argument(
        "--prompt",
        default="Retrieve the visual evidence that best answers the query.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    embedder = QwenMultimodalEmbedder(
        args.embedding_model,
        trust_remote_code=args.trust_remote_code,
    )
    image_inputs = [str(path) for path in args.images]
    document_vectors = embedder.encode(image_inputs)
    query_vector = embedder.encode([args.query], prompt=args.prompt)[0]

    index = InMemoryVectorIndex()
    index.add(document_vectors, args.images)
    results = index.search(query_vector, top_k=min(args.top_k, len(args.images)))

    print(f"Embedding model: {args.embedding_model}")
    for result in results:
        print(f"{result.rank}. score={result.score:.4f}  {result.item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
