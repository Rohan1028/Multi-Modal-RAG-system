"""Prompt templates for grounded answers."""
from __future__ import annotations

from typing import List

SYSTEM_PROMPT = (
    "You are Aurora, an expert analyst. Answer using provided context only. "
    "Cite every factual sentence with [source_id] markers."
)


def build_user_prompt(question: str, contexts: List[str]) -> str:
    """Render a user prompt that enumerates the retrieved evidence."""
    evidence = "\n\n".join(f"Source {idx+1}: {chunk}" for idx, chunk in enumerate(contexts))
    return (
        "Question: "
        + question
        + "\n\nContext:\n"
        + evidence
        + "\n\nInstructions: Provide a concise answer with inline [source_i] citations."
    )
