"""I/O helper utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

from PIL import Image


def iter_files(root: Path, extensions: Iterable[str]) -> Iterable[Path]:
    """Yield files beneath ``root`` matching one of the provided extensions."""
    normalized = {ext.lower() for ext in extensions}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in normalized:
            yield path


def load_text(path: Path) -> str:
    """Read the contents of a text file as UTF-8."""
    return path.read_text(encoding="utf-8")


def save_json(path: Path, payload: Any) -> None:
    """Persist JSON data with indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    """Load JSON from disk if the path exists, otherwise return ``None``."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_image(path: Path) -> Image.Image:
    """Load an image using Pillow."""
    return Image.open(path).convert("RGB")


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_relative(paths: Iterable[Path], root: Path) -> List[str]:
    """Return POSIX-style relative paths."""
    result: List[str] = []
    for path in paths:
        result.append(path.relative_to(root).as_posix())
    return result
