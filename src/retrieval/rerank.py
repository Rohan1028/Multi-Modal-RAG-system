"""Cross-encoder re-ranking with fallback."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from src.retrieval.embeddings import TextEmbedder, cosine_similarity
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

try:  # pragma: no cover
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None  # type: ignore


@dataclass(slots=True)
class RankedCandidate:
    content: str
    score: float
    metadata: dict
    source_id: str


class CrossEncoderReranker:
    """Applies cross-encoder scoring to ranked documents."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self._model = self._load_model()
        self._fallback = TextEmbedder()

    @staticmethod
    def _load_model():  # pragma: no cover
        if CrossEncoder is None:
            LOGGER.warning("CrossEncoder unavailable; using embedding fallback")
            return None
        try:
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as exc:
            LOGGER.warning("CrossEncoder load failed (%s); using fallback", exc)
            return None

    def rerank(self, query: str, candidates: Iterable[RankedCandidate], top_k: int) -> List[RankedCandidate]:
        cand_list = list(candidates)
        if not cand_list:
            return []
        model = self._model
        if model is None:
            query_vec = self._fallback.encode([query])[0]
            doc_vecs = self._fallback.encode([cand.content for cand in cand_list])
            scores = cosine_similarity(query_vec, doc_vecs)
            for cand, score in zip(cand_list, scores):
                cand.score = float(score)
            return sorted(cand_list, key=lambda c: c.score, reverse=True)[:top_k]

        pairs = [(query, cand.content) for cand in cand_list]
        scores = model.predict(pairs)
        for cand, score in zip(cand_list, scores):
            cand.score = float(score)
        cand_list.sort(key=lambda c: c.score, reverse=True)
        return cand_list[:top_k]
