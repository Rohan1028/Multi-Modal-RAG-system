"""Index builder orchestrating ingestion and persistence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import duckdb
import numpy as np

from src.generation.cite_guard import canonical_source_id
from src.ingestion.loaders import (
    ImageDocument,
    RawDocument,
    TableDocument,
    iter_image_documents,
    iter_table_documents,
    iter_text_documents,
    summarize_modalities,
)
from src.ingestion.preprocess import TextChunk, chunk_corpus
from src.retrieval import bm25
from src.retrieval.embeddings import ImageEmbedder, TextEmbedder
from src.retrieval.tables import TableRegistry
from src.utils.io import ensure_dir, save_json
from src.utils.logging import get_logger

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - fallback when faiss unavailable
    faiss = None

LOGGER = get_logger(__name__)


def _persist_faiss(index_path: Path, vectors: np.ndarray) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(index_path.with_suffix(".npy"), vectors)
    if faiss is None:
        LOGGER.warning("faiss not installed; saved vectors to %s for in-memory loading", index_path)
        return
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    faiss.write_index(index, str(index_path))


def _summaries_from_chunks(chunks: List[TextChunk]) -> List[Dict[str, object]]:
    return [
        {
            "content": chunk.content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]


def build_indexes(data_root: Path, index_dir: Path, duckdb_path: Path) -> Dict[str, object]:
    """Main entry point for indexing data under ``data_root``."""
    ensure_dir(index_dir)
    ensure_dir(duckdb_path.parent)

    conn = duckdb.connect(str(duckdb_path))
    text_docs = list(iter_text_documents(data_root / "docs"))
    image_docs = list(iter_image_documents(data_root / "images"))
    table_docs = list(iter_table_documents(data_root / "tables", conn))
    modality_summary = summarize_modalities(text_docs, image_docs, table_docs)
    LOGGER.info("Loaded modalities: %s", modality_summary)

    # Text processing
    text_chunks = chunk_corpus(text_docs)
    text_encoder = TextEmbedder()
    text_vectors = text_encoder.encode([chunk.content for chunk in text_chunks])
    if len(text_vectors):
        _persist_faiss(index_dir / "text.index", text_vectors)
        save_json(index_dir / "text_metadata.json", _summaries_from_chunks(text_chunks))
        bm25.build_index(index_dir / "bm25", text_chunks)

    # Image processing
    image_encoder = ImageEmbedder()
    image_vectors = image_encoder.encode_paths([img.path for img in image_docs])
    if len(image_vectors):
        _persist_faiss(index_dir / "image.index", image_vectors)
        save_json(
            index_dir / "image_metadata.json",
            [
                {
                    "source_id": canonical_source_id(idx),
                    "metadata": doc.metadata,
                }
                for idx, doc in enumerate(image_docs, start=1)
            ],
        )

    # Tables
    registry = TableRegistry(connection=conn, index_dir=index_dir)
    registry.register_tables(table_docs)

    summary = {
        "texts_indexed": len(text_chunks),
        "images_indexed": len(image_docs),
        "tables_indexed": len(table_docs),
        "index_dir": index_dir.as_posix(),
    }
    LOGGER.info("Index build summary: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build indexes for the multimodal RAG system")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--index-dir", type=Path, default=Path(".cache/index"))
    parser.add_argument("--duckdb-path", type=Path, default=Path(".cache/duckdb/catalog.db"))
    args = parser.parse_args()

    build_indexes(data_root=args.data_root, index_dir=args.index_dir, duckdb_path=args.duckdb_path)


if __name__ == "__main__":
    main()

