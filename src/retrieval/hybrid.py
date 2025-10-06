"""Hybrid retrieval combining dense, sparse, image, and table signals."""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

from src.generation.cite_guard import canonical_source_id
from src.retrieval import bm25
from src.retrieval.embeddings import ImageEmbedder, TextEmbedder, cosine_similarity
from src.retrieval.rerank import CrossEncoderReranker, RankedCandidate
from src.retrieval.tables import TableRegistry
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

DEFAULT_TOP_K = 5
RRF_K = 60


@dataclass(slots=True)
class Context:
    source_id: str
    content: str
    metadata: Dict[str, object]
    modality: str
    score: float


class HybridRetriever:
    """Coordinate multiple retrieval strategies for multimodal queries."""

    def __init__(self, index_dir: Path, duckdb_path: Path) -> None:
        self.index_dir = index_dir
        self.duckdb_path = duckdb_path
        self.text_embedder = TextEmbedder()
        self.image_embedder = ImageEmbedder()
        self.reranker = CrossEncoderReranker()
        self.table_registry = TableRegistry(connection=self._connect_duckdb(), index_dir=index_dir)
        self.text_vectors = self._load_vectors("text.index.npy")
        self.image_vectors = self._load_vectors("image.index.npy")
        self.text_metadata = self._load_json("text_metadata.json")
        self.image_metadata = self._load_json("image_metadata.json")

    def _connect_duckdb(self):  # pragma: no cover - simple pass-through
        import duckdb

        return duckdb.connect(str(self.duckdb_path), read_only=False)

    def _load_vectors(self, filename: str) -> np.ndarray:
        path = self.index_dir / filename
        if not path.exists():
            return np.zeros((0, 1), dtype=np.float32)
        return np.load(path)

    def _load_json(self, filename: str):
        path = self.index_dir / filename
        if not path.exists():
            return []
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    def _rrf(self, scored_lists: Iterable[List[Context]]) -> List[Context]:
        accumulator: Dict[str, Context] = {}
        for contexts in scored_lists:
            for rank, ctx in enumerate(contexts, start=1):
                weight = 1.0 / (RRF_K + rank)
                if ctx.source_id not in accumulator:
                    accumulator[ctx.source_id] = ctx
                    accumulator[ctx.source_id].score = 0.0
                accumulator[ctx.source_id].score += weight
        return sorted(accumulator.values(), key=lambda c: c.score, reverse=True)

    def _dense_text(self, query: str, top_k: int) -> List[Context]:
        if self.text_vectors.size == 0 or not self.text_metadata:
            return []
        query_vec = self.text_embedder.encode([query])[0]
        scores = cosine_similarity(query_vec, self.text_vectors)
        top_indices = np.argsort(scores)[::-1][:top_k]
        contexts: List[Context] = []
        for idx in top_indices:
            meta = self.text_metadata[int(idx)]
            source_id = meta["metadata"].get("chunk_id") or f"text_{idx}"
            contexts.append(
                Context(
                    source_id=str(source_id),
                    content=meta["content"],
                    metadata=meta["metadata"],
                    modality="text",
                    score=float(scores[idx]),
                )
            )
        return contexts

    def _sparse_text(self, query: str, top_k: int) -> List[Context]:
        results = bm25.search(self.index_dir / "bm25", query, top_k=top_k)
        contexts: List[Context] = []
        for idx, item in enumerate(results):
            source_id = item["metadata"].get("chunk_id") or f"bm25_{idx}"
            contexts.append(
                Context(
                    source_id=str(source_id),
                    content=item["content"],
                    metadata=item["metadata"],
                    modality="text",
                    score=float(item["score"]),
                )
            )
        return contexts

    def _image_bridge(self, image_b64: Optional[str], top_k: int) -> List[Context]:
        if not image_b64 or self.image_vectors.size == 0 or not self.image_metadata:
            return []
        image = self._load_uploaded_image(image_b64)
        vec = self.image_embedder.encode_arrays([image])[0]
        image.close()
        scores = cosine_similarity(vec, self.image_vectors)
        top_indices = np.argsort(scores)[::-1][:top_k]
        contexts: List[Context] = []
        for idx in top_indices:
            meta = self.image_metadata[int(idx)]
            contexts.append(
                Context(
                    source_id=meta.get("source_id", f"image_{idx}"),
                    content="Relevant image evidence",
                    metadata=meta.get("metadata", {}),
                    modality="image",
                    score=float(scores[idx]),
                )
            )
        return contexts

    def _load_uploaded_image(self, image_b64: str) -> Image.Image:
        data = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(data)).convert("RGB")

    def _table_results(self, query: str, top_k: int) -> List[Context]:
        table_hits = self.table_registry.search_cards(query, top_k=top_k)
        contexts: List[Context] = []
        for table in table_hits:
            sql = self.table_registry.keyword_to_sql(query, [table.table])
            rows: List[Dict[str, object]] = []
            if sql:
                try:
                    rows = self.table_registry.run_sql(sql)
                except Exception as exc:  # pragma: no cover
                    LOGGER.warning("DuckDB query failed for %s: %s", table.table, exc)
            snippet = table.snippet
            if rows:
                sample = rows[0]
                snippet += "\nSample row: " + ", ".join(f"{k}={v}" for k, v in sample.items())
            contexts.append(
                Context(
                    source_id=f"table_{table.table}",
                    content=snippet,
                    metadata=table.metadata,
                    modality="table",
                    score=table.score,
                )
            )
        return contexts

    def retrieve(self, query_text: str, top_k: int = DEFAULT_TOP_K, image_b64: Optional[str] = None) -> Dict[str, object]:
        dense = self._dense_text(query_text, top_k * 3)
        sparse = self._sparse_text(query_text, top_k * 3)
        tables = self._table_results(query_text, top_k)
        images = self._image_bridge(image_b64, top_k)

        fused = self._rrf([dense, sparse, images, tables])
        ranked_candidates = [
            RankedCandidate(
                content=ctx.content,
                score=ctx.score,
                metadata=ctx.metadata,
                source_id=ctx.source_id,
            )
            for ctx in fused
        ]
        reranked = self.reranker.rerank(query_text, ranked_candidates, top_k=top_k)
        final_contexts: List[Context] = []
        modality_counts: Dict[str, int] = {"text": 0, "image": 0, "table": 0}
        for idx, candidate in enumerate(reranked):
            source_id = canonical_source_id(idx)
            meta = dict(candidate.metadata)
            meta.setdefault("original_source_id", candidate.source_id)
            modality = meta.get("modality", "text")
            modality_counts[modality] = modality_counts.get(modality, 0) + 1
            meta["retrieval_rank"] = idx + 1
            final_contexts.append(
                Context(
                    source_id=source_id,
                    content=candidate.content,
                    metadata=meta,
                    modality=modality,
                    score=candidate.score,
                )
            )
        return {
            "contexts": final_contexts,
            "modality_breakdown": modality_counts,
        }
