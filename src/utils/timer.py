"""Simple timing utilities."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict, Generator


@contextmanager
def track_time(output: Dict[str, float], key: str) -> Generator[None, None, None]:
    """Context manager that measures elapsed seconds and stores them in ``output``."""
    start = time.perf_counter()
    try:
        yield
    finally:
        output[key] = round((time.perf_counter() - start) * 1000, 2)
