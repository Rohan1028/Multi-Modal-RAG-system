"""Generative layer for grounded answers."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, List

from src.generation import prompt
from src.generation.cite_guard import fill_missing_sources, validate
from src.retrieval.hybrid import Context
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


@dataclass
class GenerationResult:
    answer: str
    citations: List[str]


class LocalGenerator:
    """Deterministic summarizer that stitches together context."""

    def generate(self, question: str, contexts: Iterable[Context]) -> GenerationResult:
        contexts_list = list(contexts)
        if not contexts_list:
            return GenerationResult(answer="I could not find relevant evidence.", citations=[])
        snippets: List[str] = []
        citations: List[str] = []
        for ctx in contexts_list[:3]:
            snippet = ctx.content.split("\n")[0]
            citation = ctx.source_id if ctx.source_id.startswith("source_") else f"[{ctx.source_id}]"
            if not citation.startswith("["):
                citation = f"[{citation}]"
            snippets.append(f"{snippet} {citation}")
            citations.append(ctx.source_id)
        answer = " ".join(snippets)
        cleaned = fill_missing_sources(answer, citations)
        return GenerationResult(answer=cleaned, citations=citations)


class OpenAIGenerator:
    """Thin wrapper around OpenAI responses."""

    def __init__(self) -> None:
        if OpenAI is None:
            raise RuntimeError("OpenAI client not available")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not configured")
        self.client = OpenAI(api_key=api_key)

    def generate(self, question: str, contexts: Iterable[Context]) -> GenerationResult:  # pragma: no cover
        contexts_list = list(contexts)
        citations = [ctx.source_id for ctx in contexts_list]
        user_prompt = prompt.build_user_prompt(question, [ctx.content for ctx in contexts_list])
        completion: Any = self.client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": prompt.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = completion.output[0].content[0].text  # type: ignore[index]
        if not validate(answer, citations):
            answer = fill_missing_sources(answer, citations)
        return GenerationResult(answer=answer, citations=citations)


def get_generator(backend: str | None = None):
    backend = (backend or "local").lower()
    if backend == "openai":
        try:
            return OpenAIGenerator()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Falling back to local generator: %s", exc)
    return LocalGenerator()
