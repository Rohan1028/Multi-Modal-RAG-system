"""FastAPI dependency wiring."""
from __future__ import annotations

from functools import lru_cache

from src.app.settings import Settings, get_settings
from src.generation.generator import get_generator
from src.retrieval.hybrid import HybridRetriever


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    settings = get_settings()
    return HybridRetriever(index_dir=settings.index_dir, duckdb_path=settings.duckdb_path)


def get_answer_generator():
    settings = get_settings()
    return get_generator(settings.generator_backend)
