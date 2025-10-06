from pathlib import Path

from src.ingestion.indexer import build_indexes


def test_build_indexes(tmp_path: Path) -> None:
    index_dir = tmp_path / "index"
    duckdb_path = tmp_path / "duckdb/catalog.db"
    summary = build_indexes(Path("data"), index_dir, duckdb_path)
    assert summary["texts_indexed"] > 0
    assert summary["images_indexed"] > 0
    assert summary["tables_indexed"] > 0
    assert (index_dir / "text_metadata.json").exists()
    assert (index_dir / "bm25").exists()
