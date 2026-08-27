# Path: src/vlm_engineering/documents/fusion.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Fuse deterministic/native text with VLM-derived visual semantics."""

from __future__ import annotations

from .schemas import NativePageContent, VisualAnalysis


def fuse_native_and_visual(native: NativePageContent, visual: VisualAnalysis) -> str:
    """Create retrieval text while keeping native text and visual semantics distinguishable."""
    parts: list[str] = []
    if native.title:
        parts.append(f"# {native.title}")
    if native.native_text.strip():
        parts.extend(["## Native text", native.native_text.strip()])
    if visual.summary:
        parts.extend(["## Visual summary", visual.summary.strip()])
    if visual.entities:
        parts.extend(["## Visual entities", ", ".join(visual.entities)])
    if visual.relations:
        parts.append("## Visual relations")
        parts.extend(
            f"- {r.source} -> {r.target}: {r.relation}" + (f" ({r.evidence})" if r.evidence else "")
            for r in visual.relations
        )
    if visual.important_text:
        parts.extend(["## Important visual text", "\n".join(f"- {t}" for t in visual.important_text)])
    if visual.uncertainties:
        parts.extend(["## Uncertainties", "\n".join(f"- {u}" for u in visual.uncertainties)])
    return "\n\n".join(part for part in parts if part).strip()
