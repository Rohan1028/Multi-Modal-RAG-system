"""Data loaders for text, image, and tabular modalities."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import duckdb
import pandas as pd
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as extract_pdf_text

from src.utils.io import load_image, load_text


@dataclass(slots=True)
class RawDocument:
    """Represents an unprocessed document."""

    path: Path
    content: str
    modality: str
    metadata: Dict[str, Any]


@dataclass(slots=True)
class ImageDocument:
    """Represents an image ready for embedding."""

    path: Path
    modality: str
    metadata: Dict[str, Any]


@dataclass(slots=True)
class TableDocument:
    """Represents a tabular asset registered in DuckDB."""

    name: str
    path: Path
    dataframe: pd.DataFrame
    card: str
    metadata: Dict[str, Any]


SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
SUPPORTED_TABLE_EXTENSIONS = {".csv", ".tsv", ".parquet"}


def load_text_document(path: Path) -> RawDocument:
    """Load a text-like document and return its raw content."""
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown"}:
        content = load_text(path)
    elif suffix in {".html", ".htm"}:
        soup = BeautifulSoup(load_text(path), "html.parser")
        content = soup.get_text(separator="\n")
    elif suffix == ".pdf":
        content = extract_pdf_text(str(path))
    else:
        raise ValueError(f"Unsupported text extension: {suffix}")

    metadata = {
        "source_path": path.as_posix(),
        "modality": "text",
    }
    return RawDocument(path=path, content=content.strip(), modality="text", metadata=metadata)


def iter_text_documents(root: Path) -> Iterable[RawDocument]:
    """Yield documents for all supported text files in ``root``."""
    for path in root.rglob("*"):
        if path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS and path.is_file():
            yield load_text_document(path)


def load_image_document(path: Path) -> ImageDocument:
    """Load metadata for an image. Actual pixel data is handled downstream."""
    image = load_image(path)
    metadata = {
        "width": image.width,
        "height": image.height,
        "source_path": path.as_posix(),
        "modality": "image",
    }
    image.close()
    return ImageDocument(path=path, modality="image", metadata=metadata)


def iter_image_documents(root: Path) -> Iterable[ImageDocument]:
    """Yield all supported images underneath ``root``."""
    for path in root.rglob("*"):
        if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and path.is_file():
            yield load_image_document(path)


def load_table_document(path: Path, connection: duckdb.DuckDBPyConnection) -> TableDocument:
    """Register the table in DuckDB and return metadata including a text card."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported table extension: {suffix}")

    table_name = path.stem.replace("-", "_").replace(" ", "_")
    connection.register(table_name, df)
    preview = df.head(3)
    card_lines = [f"Table {table_name}"]
    card_lines.append("Columns: " + ", ".join(str(col) for col in df.columns))
    card_lines.append("Sample rows:")
    for _, row in preview.iterrows():
        card_lines.append(" - " + ", ".join(f"{col}={row[col]}" for col in df.columns))
    card = "\n".join(card_lines)
    metadata = {
        "rows": len(df),
        "columns": list(df.columns),
        "source_path": path.as_posix(),
        "modality": "table",
    }
    return TableDocument(name=table_name, path=path, dataframe=df, card=card, metadata=metadata)


def iter_table_documents(root: Path, connection: duckdb.DuckDBPyConnection) -> Iterable[TableDocument]:
    """Yield tables ready for indexing."""
    for path in root.rglob("*"):
        if path.suffix.lower() in SUPPORTED_TABLE_EXTENSIONS and path.is_file():
            yield load_table_document(path, connection)


def summarize_modalities(text_docs: List[RawDocument], image_docs: List[ImageDocument], table_docs: List[TableDocument]) -> Dict[str, int]:
    """Return a high-level summary for logging purposes."""
    return {
        "text_documents": len(text_docs),
        "image_documents": len(image_docs),
        "table_documents": len(table_docs),
    }
