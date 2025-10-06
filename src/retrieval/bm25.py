"""BM25 index using Whoosh with a lightweight fallback."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

from src.ingestion.preprocess import TextChunk
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

try:  # pragma: no cover
    from whoosh import index
    from whoosh.fields import ID, TEXT, Schema
    from whoosh.qparser import MultifieldParser
except Exception:  # pragma: no cover
    index = None  # type: ignore
    Schema = None  # type: ignore


FALLBACK_FILE = "bm25_fallback.json"


def build_index(index_dir: Path, chunks: Iterable[TextChunk]) -> None:
    """Create or rebuild a BM25 index."""
    ensure_dir(index_dir)
    chunk_list = list(chunks)
    if index is None:
        LOGGER.warning("Whoosh unavailable; storing BM25 fallback JSON")
        payload = [
            {
                "doc_id": str(idx),
                "content": chunk.content,
                "metadata": chunk.metadata,
            }
            for idx, chunk in enumerate(chunk_list)
        ]
        (index_dir / FALLBACK_FILE).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return

    schema = Schema(doc_id=ID(stored=True, unique=True), content=TEXT(stored=True))
    index_dir.mkdir(parents=True, exist_ok=True)
    if index.exists_in(index_dir):
        for child in Path(index_dir).iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    idx = index.create_in(index_dir, schema)
    writer = idx.writer()
    for doc_id, chunk in enumerate(chunk_list):
        writer.add_document(doc_id=str(doc_id), content=chunk.content)
    writer.commit()

    # Persist metadata separately for retrieval
    metadata_payload = [
        {
            "doc_id": str(doc_id),
            "content": chunk.content,
            "metadata": chunk.metadata,
        }
        for doc_id, chunk in enumerate(chunk_list)
    ]
    (index_dir / "metadata.json").write_text(json.dumps(metadata_payload, ensure_ascii=False), encoding="utf-8")


def search(index_dir: Path, query: str, top_k: int = 5) -> List[Dict[str, object]]:
    """Execute a BM25 search returning scored documents."""
    if index is None or not index.exists_in(index_dir):
        # Fallback path using JSON similarity
        fallback_path = index_dir / FALLBACK_FILE
        if not fallback_path.exists():
            return []
        payload = json.loads(fallback_path.read_text(encoding="utf-8"))
        terms = set(query.lower().split())
        scored: List[Dict[str, object]] = []
        for item in payload:
            text = item["content"].lower()
            overlap = sum(1 for term in terms if term in text)
            if overlap:
                scored.append({
                    "score": float(overlap),
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "doc_id": item["doc_id"],
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    idx = index.open_dir(index_dir)
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    searcher = idx.searcher()
    parser = MultifieldParser(["content"], schema=idx.schema)
    query_obj = parser.parse(query)
    results = searcher.search(query_obj, limit=top_k)
    output: List[Dict[str, object]] = []
    for hit in results:
        doc_id = hit["doc_id"]
        meta = next((item for item in metadata if item["doc_id"] == doc_id), None)
        if meta:
            output.append({
                "score": float(hit.score),
                "content": meta["content"],
                "metadata": meta["metadata"],
                "doc_id": doc_id,
            })
    searcher.close()
    return output


