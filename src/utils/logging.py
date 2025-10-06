"""Logging utilities for the multimodal RAG system."""
from __future__ import annotations

import logging
from logging import Logger
from typing import Optional

_LOGGERS: dict[str, Logger] = {}


def get_logger(name: str = "multimodal_rag") -> Logger:
    """Return a module-level logger configured for structured output."""
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.propagate = False
    _LOGGERS[name] = logger
    return logger


def set_level(level: int, name: Optional[str] = None) -> None:
    """Update the logging level globally or for a specific named logger."""
    if name is None:
        logging.basicConfig(level=level)
    else:
        get_logger(name).setLevel(level)
