"""Citation guard utilities to enforce grounded outputs."""
from __future__ import annotations

import re
from typing import Iterable, List, Sequence

SOURCE_PATTERN = re.compile(r"\[(source_\d+)\]")


def canonical_source_id(index: int) -> str:
    return f"source_{index + 1}"


def extract_source_ids(text: str) -> List[str]:
    return SOURCE_PATTERN.findall(text)


def sentences(text: str) -> List[str]:
    """Split text into sentences while keeping trailing citation tokens attached."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences: List[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("[") and sentences:
            sentences[-1] = f"{sentences[-1]} {part}".strip()
        else:
            sentences.append(part.strip())
    return sentences


def validate(answer: str, available_ids: Sequence[str]) -> bool:
    if not answer.strip():
        return False
    available = set(available_ids)
    for sentence in sentences(answer):
        ids = set(extract_source_ids(sentence))
        if not ids:
            return False
        if not ids.issubset(available):
            return False
    return True


def fill_missing_sources(answer: str, available_ids: Sequence[str]) -> str:
    """Append a Sources footer if sentences missed citations."""
    if validate(answer, available_ids):
        return answer
    footer = "\n\nSources: " + ", ".join(f"[{sid}]" for sid in available_ids)
    return answer + footer


def filter_contexts(contexts: Iterable[dict]) -> List[str]:
    chunks: List[str] = []
    for ctx in contexts:
        content = ctx.get("content") if isinstance(ctx, dict) else getattr(ctx, "content", "")
        if content:
            chunks.append(str(content))
    return chunks
