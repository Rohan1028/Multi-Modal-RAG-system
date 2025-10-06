"""Embedding utilities with graceful fallbacks."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Sequence, cast

import numpy as np
from PIL import Image
from numpy.typing import NDArray

try:  # pragma: no cover - heavy dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:  # pragma: no cover - optional
    from transformers import CLIPModel, CLIPProcessor
    import torch
except Exception:  # pragma: no cover
    CLIPModel = None  # type: ignore
    CLIPProcessor = None  # type: ignore
    torch = None  # type: ignore

from src.utils.io import load_image
from src.utils.logging import get_logger

LOGGER = get_logger(__name__)

FloatArray = NDArray[np.float32]


def _hash_to_vector(payload: str, dim: int = 384) -> FloatArray:
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    expanded = (digest * ((dim // len(digest)) + 1))[:dim]
    arr = np.frombuffer(expanded, dtype=np.uint8).astype(np.float32)
    norm = np.linalg.norm(arr) or 1.0
    return cast(FloatArray, arr / norm)


class TextEmbedder:
    """Wraps sentence-transformers with a deterministic fallback."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = self._load_model()
        self._dim = 384

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model():  # pragma: no cover - exercised indirectly
        if SentenceTransformer is None:
            LOGGER.warning("SentenceTransformer unavailable; using hash-based embeddings")
            return None
        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as exc:  # downloads might fail offline
            LOGGER.warning("Falling back to hash embeddings: %s", exc)
            return None

    def encode(self, texts: Sequence[str]) -> FloatArray:
        if not texts:
            return cast(FloatArray, np.zeros((0, self._dim), dtype=np.float32))
        model = self._model
        if model is None:
            vectors = np.stack([_hash_to_vector(text, self._dim) for text in texts])
        else:
            encoded = model.encode(list(texts), normalize_embeddings=True)
            vectors = np.asarray(encoded, dtype=np.float32)
        return cast(FloatArray, vectors)


class ImageEmbedder:
    """CLIP-based image encoder with fallback."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        self.model_name = model_name
        self._model, self._processor = self._load_model()
        self._dim = 512

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model():  # pragma: no cover
        if CLIPModel is None or CLIPProcessor is None or torch is None:
            LOGGER.warning("CLIP unavailable; using pixel-average embeddings")
            return None, None
        try:
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model.eval()
            return model, processor
        except Exception as exc:
            LOGGER.warning("CLIP load failed (%s); using fallback", exc)
            return None, None

    def encode_paths(self, image_paths: Sequence[Path]) -> FloatArray:
        if not image_paths:
            return cast(FloatArray, np.zeros((0, self._dim), dtype=np.float32))
        model, processor = self._model, self._processor
        if model is None or processor is None:
            vectors: List[FloatArray] = []
            for path in image_paths:
                image = load_image(path)
                arr = np.array(image).astype(np.float32)
                vec = arr.mean(axis=(0, 1))
                padded = np.pad(vec, (0, max(0, self._dim - vec.size)))[: self._dim]
                norm = np.linalg.norm(padded) or 1.0
                normalized = (padded / norm).astype(np.float32)
                vectors.append(cast(FloatArray, normalized))
                image.close()
            return cast(FloatArray, np.stack(vectors))

        images: List[Image.Image] = [load_image(path) for path in image_paths]
        inputs = processor(images=images, return_tensors="pt")
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
        for image in images:
            image.close()
        return cast(FloatArray, features.cpu().numpy().astype(np.float32))

    def encode_arrays(self, images: Iterable[Image.Image]) -> FloatArray:
        """Encode in-memory images by writing temporary files."""
        import tempfile

        temp_paths: List[Path] = []
        try:
            for image in images:
                temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                image.save(temp_file.name)
                temp_file.close()
                temp_paths.append(Path(temp_file.name))
            return self.encode_paths(tuple(temp_paths))
        finally:
            for path in temp_paths:
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue


def cosine_similarity(query: FloatArray, matrix: FloatArray) -> FloatArray:
    if matrix.size == 0:
        return cast(FloatArray, np.zeros((0,), dtype=np.float32))
    query_norm = query / (np.linalg.norm(query) or 1.0)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return cast(FloatArray, matrix_norm @ query_norm)
