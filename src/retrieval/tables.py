"""DuckDB-backed table retrieval helpers."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import duckdb
import numpy as np
import pandas as pd

from src.ingestion.loaders import TableDocument
from src.retrieval.embeddings import TextEmbedder, cosine_similarity
from src.utils.io import ensure_dir
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

CATALOG_FILE = "table_catalog.json"
EMBED_FILE = "table_embeddings.npy"


@dataclass(slots=True)
class TableResult:
    table: str
    snippet: str
    score: float
    metadata: Dict[str, Any]


class TableRegistry:
    """Manage DuckDB registrations and metadata."""

    def __init__(self, connection: duckdb.DuckDBPyConnection, index_dir: Path) -> None:
        self.connection = connection
        self.index_dir = ensure_dir(index_dir)
        self.catalog_path = self.index_dir / CATALOG_FILE
        self.embed_path = self.index_dir / EMBED_FILE
        self.embedder = TextEmbedder()

    def register_tables(self, tables: Iterable[TableDocument]) -> None:
        catalog: List[Dict[str, Any]] = []
        cards: List[str] = []
        for table in tables:
            catalog.append(
                {
                    "name": table.name,
                    "card": table.card,
                    "metadata": table.metadata,
                }
            )
            cards.append(table.card)
        if catalog:
            self.catalog_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
            vectors = self.embedder.encode(cards)
            np.save(self.embed_path, vectors)
        LOGGER.info("Registered %s tables", len(catalog))

    def _load_catalog(self) -> List[Dict[str, Any]]:
        if not self.catalog_path.exists():
            return []
        return json.loads(self.catalog_path.read_text(encoding="utf-8"))

    def ensure_table_loaded(self, entry: Dict[str, Any]) -> None:
        name = entry["name"]
        metadata = entry.get("metadata", {})
        source_path = metadata.get("source_path")
        if source_path is None:
            return
        # If table already exists in connection, skip
        existing = self.connection.execute("SELECT table_name FROM information_schema.tables").fetchall()
        if any(row[0] == name for row in existing):
            return
        path = Path(source_path)
        if not path.exists():
            LOGGER.warning("Table path missing: %s", path)
            return
        if path.suffix.lower() in {".csv", ".tsv"}:
            df = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
        else:
            LOGGER.warning("Unsupported table extension for %s", path)
            return
        self.connection.register(name, df)

    def search_cards(self, query: str, top_k: int = 3) -> List[TableResult]:
        catalog = self._load_catalog()
        if not catalog or not self.embed_path.exists():
            return []
        embeddings = np.load(self.embed_path)
        query_vec = self.embedder.encode([query])[0]
        scores = cosine_similarity(query_vec, embeddings)
        ranked = np.argsort(scores)[::-1][:top_k]
        results: List[TableResult] = []
        for idx in ranked:
            if scores[idx] <= 0:
                continue
            entry = catalog[int(idx)]
            self.ensure_table_loaded(entry)
            results.append(
                TableResult(
                    table=entry["name"],
                    snippet=entry["card"],
                    score=float(scores[idx]),
                    metadata=entry["metadata"],
                )
            )
        return results

    def keyword_to_sql(self, query: str, candidates: Sequence[str]) -> str | None:
        """Very small heuristic to produce SQL from keywords."""
        words = [kw for kw in re.findall(r"[a-zA-Z0-9_]+", query.lower()) if len(kw) > 2]
        if not words:
            return None
        catalog = self._load_catalog()
        if not catalog:
            return None
        table_lookup: Dict[str, Dict[str, Any]] = {entry["name"]: entry for entry in catalog}
        target_table = candidates[0] if candidates else catalog[0]["name"]
        entry = table_lookup.get(target_table)
        if entry is None:
            return None
        self.ensure_table_loaded(entry)
        columns: List[str] = entry["metadata"].get("columns", [])
        filters = [kw for kw in words if kw not in {"show", "list", "table", "data", "report"}]
        where_clause = ""
        if filters and columns:
            like = []
            for column in columns:
                like.append(f"LOWER({column}) LIKE '%{filters[0]}%'")
            where_clause = " WHERE " + " OR ".join(like)
        sql = f"SELECT * FROM {target_table}{where_clause} LIMIT 5"
        return sql

    def run_sql(self, sql: str) -> List[Dict[str, Any]]:
        result = self.connection.execute(sql).fetchdf()
        return result.to_dict(orient="records")
