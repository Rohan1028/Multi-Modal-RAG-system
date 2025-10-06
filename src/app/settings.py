"""Application settings."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    index_dir: Path = Field(default=Path(".cache/index"))
    duckdb_path: Path = Field(default=Path(".cache/duckdb/catalog.db"))
    generator_backend: str = Field(default="local")

    model_config = {
        "env_prefix": "MMS_",
        "env_file": ".env",
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.index_dir.parent.mkdir(parents=True, exist_ok=True)
    settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    return settings
