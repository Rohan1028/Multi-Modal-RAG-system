"""Pre-processing utilities for ingestion."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List

from src.ingestion.loaders import RawDocument


@dataclass(slots=True)
class TextChunk:
    """A normalized chunk of text with metadata."""

    content: str
    metadata: Dict[str, object]


def _tokenize(text: str) -> List[str]:
    return text.split()


def chunk_text(
    document: RawDocument,
    chunk_size: int = 220,
    overlap: int = 40,
) -> List[TextChunk]:
    """Split a document into overlapping chunks measured in tokens."""
    tokens = _tokenize(document.content)
    if not tokens:
        return []

    chunks: List[TextChunk] = []
    step = max(chunk_size - overlap, 1)
    total_chunks = math.ceil(len(tokens) / step)
    for idx in range(total_chunks):
        start = idx * step
        end = min(start + chunk_size, len(tokens))
        window = tokens[start:end]
        if not window:
            continue
        chunk_text_value = " ".join(window)
        metadata = dict(document.metadata)
        metadata.update({
            "chunk_id": f"{document.path.name}-chunk-{idx}",
            "chunk_index": idx,
            "token_start": start,
            "token_end": end,
        })
        chunks.append(TextChunk(content=chunk_text_value, metadata=metadata))
    return chunks


def chunk_corpus(documents: Iterable[RawDocument]) -> List[TextChunk]:
    """Chunk a collection of documents."""
    results: List[TextChunk] = []
    for doc in documents:
        results.extend(chunk_text(doc))
    return results


def ocr_fallback(path: str) -> str:
    """Placeholder OCR fallback."""
    # Real implementation would call pytesseract or another OCR tool. We log a warning upstream.
    return ""
