# Path: tests/unit/test_prompt_templates.py
# Author: GHANMI Helmi
# Current Role: AI Engineer
# Past Role: Researcher in Applied Mathematics
# Research Profile: https://www.researchgate.net/profile/Ghanmi-Helmi

"""Tests for prompt templates."""

from vlm_engineering.prompt_templates.templates import (
    DIAGRAM_PROMPT,
    PAGE_TYPE_PROMPTS,
    prompt_for_page_type,
    rag_answer_prompt,
)


def test_page_type_prompts_cover_all_types() -> None:
    for page_type in ("diagram", "table", "text", "screenshot", "photo", "mixed"):
        assert page_type in PAGE_TYPE_PROMPTS
        assert PAGE_TYPE_PROMPTS[page_type].strip()


def test_prompt_for_page_type_returns_diagram_by_default() -> None:
    prompt = prompt_for_page_type("diagram")
    assert "diagram" in prompt.lower()
    assert "JSON" in prompt


def test_prompt_for_page_type_falls_back_to_mixed() -> None:
    default = prompt_for_page_type("mixed")
    unknown = prompt_for_page_type("not-a-real-type")
    assert unknown == default


def test_rag_answer_prompt_includes_evidence_and_question() -> None:
    prompt = rag_answer_prompt("what is cached?", "SOURCE: f.pdf, page 1\nRedis caches.")
    assert "what is cached?" in prompt
    assert "Redis caches." in prompt


def test_rag_answer_prompt_marks_query_image() -> None:
    with_query = rag_answer_prompt("q", "evidence", query_image=True)
    without_query = rag_answer_prompt("q", "evidence", query_image=False)
    assert "Image 1 is the query image" in with_query
    assert "Image 1 is the query image" not in without_query


def test_diagram_prompt_is_exported() -> None:
    assert DIAGRAM_PROMPT.strip()
    assert "entities" in DIAGRAM_PROMPT
