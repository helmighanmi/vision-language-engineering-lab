# Path: src/vlm_engineering/cli.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Command-line interface for CLIP, Qwen3-VL and visual document workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .documents.chunking import VisualChunkBuilder
from .documents.schemas import NativePageContent
from .documents.visual_analysis import analyze_visual_document
from .qwen.downloader import download_model_snapshot
from .qwen.model import QwenVLModel
from .qwen.registry import (
    DEFAULT_QWEN_MODEL_SIZE,
    QWEN3_VL_INSTRUCT_MODELS,
    default_model_directory,
    resolve_qwen_model_id,
)


def _build_qwen(args: argparse.Namespace) -> QwenVLModel:
    """Build a Qwen wrapper from local path, explicit Hub ID, or size preset."""
    if getattr(args, "model_path", None):
        return QwenVLModel.from_local(
            args.model_path,
            trust_remote_code=args.trust_remote_code,
        )

    if getattr(args, "model_id", None):
        return QwenVLModel.from_hub(
            args.model_id,
            trust_remote_code=args.trust_remote_code,
        )

    return QwenVLModel.from_preset(
        getattr(args, "model_size", None) or DEFAULT_QWEN_MODEL_SIZE,
        trust_remote_code=args.trust_remote_code,
    )


def cmd_describe(args: argparse.Namespace) -> int:
    """Describe or answer a question about one image."""
    model = _build_qwen(args)
    print(
        model.generate(
            args.image,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
        )
    )
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Produce structured visual-analysis JSON for one image."""
    model = _build_qwen(args)
    analysis = analyze_visual_document(model, args.image)
    print(json.dumps(analysis.to_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    """Build a semantic visual RAG chunk."""
    model = _build_qwen(args)
    analysis = analyze_visual_document(model, args.image)

    native = NativePageContent(
        document_id=args.document_id,
        page=args.page,
        source_file=args.source_file,
        native_text=(
            Path(args.native_text_file).read_text(encoding="utf-8")
            if args.native_text_file
            else ""
        ),
        image_ref=args.image,
    )

    chunk = VisualChunkBuilder().build(
        native,
        analysis,
        parent_context=args.parent_context,
    )

    print(json.dumps(chunk.to_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    """Download one Qwen snapshot into an explicit project-local directory."""
    model_id = resolve_qwen_model_id(
        model_size=args.model_size,
        model_id=args.model_id,
    )

    output = Path(args.output) if args.output else default_model_directory(model_id)

    path = download_model_snapshot(
        model_id,
        output_dir=output,
        revision=args.revision,
    )

    print(path)
    return 0


def cmd_models(_args: argparse.Namespace) -> int:
    """List supported friendly Qwen3-VL model presets."""
    print("Qwen3-VL Instruct presets:\n")

    for size, preset in QWEN3_VL_INSTRUCT_MODELS.items():
        default_marker = " (default)" if size == DEFAULT_QWEN_MODEL_SIZE else ""
        print(
            f"{size}{default_marker}: {preset.model_id} "
            f"[{preset.parameter_class}]\n  {preset.description}"
        )

    return 0


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add mutually exclusive Qwen model-selection arguments."""
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--model-size",
        choices=tuple(QWEN3_VL_INSTRUCT_MODELS),
        help="Friendly Qwen3-VL Instruct preset: 2b (default), 4b, or 8b.",
    )

    group.add_argument(
        "--model-id",
        help="Explicit compatible Hugging Face model ID; overrides the default preset.",
    )

    group.add_argument(
        "--model-path",
        help=(
            "Explicit local model directory. This enables local-files-only loading "
            "and is the recommended option for offline/Docker deployments."
        ),
    )

    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help=(
            "Opt in to executing custom model repository code. "
            "Not required for current Qwen3-VL models."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(
        prog="vlm-lab",
        description="Vision-Language Engineering Lab CLI",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    models = sub.add_parser(
        "models",
        help="List supported Qwen3-VL model-size presets.",
    )
    models.set_defaults(func=cmd_models)

    describe = sub.add_parser(
        "describe",
        help="Describe or question an image with Qwen3-VL.",
    )
    describe.add_argument("image")
    describe.add_argument(
        "--prompt",
        default="Describe this image precisely.",
    )
    describe.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
    )
    add_model_args(describe)
    describe.set_defaults(func=cmd_describe)

    analyze = sub.add_parser(
        "analyze",
        help="Extract structured JSON for RAG from an image.",
    )
    analyze.add_argument("image")
    add_model_args(analyze)
    analyze.set_defaults(func=cmd_analyze)

    chunk = sub.add_parser(
        "chunk",
        help="Build a visual RAG chunk from an image and optional native text.",
    )
    chunk.add_argument("image")
    chunk.add_argument("--document-id", required=True)
    chunk.add_argument("--source-file", required=True)
    chunk.add_argument("--page", type=int, required=True)
    chunk.add_argument("--native-text-file")
    chunk.add_argument("--parent-context", default="")
    add_model_args(chunk)
    chunk.set_defaults(func=cmd_chunk)

    download = sub.add_parser(
        "download-model",
        help=(
            "Download a complete Hugging Face model snapshot into a project-local "
            "directory for predictable local/offline use."
        ),
    )

    download_group = download.add_mutually_exclusive_group()

    download_group.add_argument(
        "--model-size",
        choices=tuple(QWEN3_VL_INSTRUCT_MODELS),
        help="Download a supported Qwen3-VL preset: 2b, 4b, or 8b.",
    )

    download_group.add_argument(
        "--model-id",
        help="Download an explicit compatible Hugging Face model ID.",
    )

    download.add_argument(
        "--output",
        help=(
            "Destination directory. Defaults to models/<model-name>; for example, "
            "models/Qwen3-VL-2B-Instruct."
        ),
    )

    download.add_argument(
        "--revision",
        help="Optional Hugging Face branch, tag, or commit revision.",
    )

    download.set_defaults(func=cmd_download)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command-line interface."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))
