from pathlib import Path

from src.ingestion.indexer import build_indexes
from src.retrieval.hybrid import HybridRetriever


def test_hybrid_retrieval(tmp_path):
    index_dir = tmp_path / ".cache/index"
    duckdb_path = tmp_path / "duckdb/catalog.db"
    build_indexes(Path("data"), index_dir, duckdb_path)

    retriever = HybridRetriever(index_dir=index_dir, duckdb_path=duckdb_path)
    result = retriever.retrieve("How did sustainability metrics change in 2024?", top_k=3)
    contexts = result["contexts"]
    assert len(contexts) >= 3
    ids = [ctx.source_id for ctx in contexts]
    assert ids == sorted(ids)
    assert all(ctx.content for ctx in contexts)
