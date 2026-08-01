"""Timing utilities for performance measurement."""

import time
import functools
import logging
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger("hermes_rag")


@contextmanager
def timer(name: str = "", log: Optional[object] = None):
    """Context manager for timing code blocks.

    Usage:
        with timer("embedding"):
            embeddings = model.encode(texts)
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"[TIMER] {name}: {elapsed:.4f}s" if name else f"[TIMER] {elapsed:.4f}s"
    target = log or logger
    target.info(msg)


def timed(func):
    """Decorator to time function execution.

    Usage:
        @timed
        def my_function():
            ...
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"[TIMER] {func.__name__}: {elapsed:.4f}s")
        return result

    return wrapper


class Stopwatch:
    """Simple stopwatch for cumulative timing."""

    def __init__(self):
        self._start = time.perf_counter()
        self._laps: list[tuple[str, float]] = []

    def lap(self, name: str) -> float:
        """Record a lap time."""
        elapsed = time.perf_counter() - self._start
        self._laps.append((name, elapsed))
        return elapsed

    def reset(self) -> None:
        """Reset the stopwatch."""
        self._start = time.perf_counter()
        self._laps.clear()

    def summary(self) -> str:
        """Return a summary of all laps."""
        lines = []
        prev = 0.0
        for name, t in self._laps:
            lines.append(f"  {name}: {t - prev:.4f}s (total: {t:.4f}s)")
            prev = t
        return "\n".join(lines)

    def durations(self) -> dict:
        """Return a dict of lap name -> delta time (seconds).

        Returns:
            dict mapping lap names to their individual durations.
        """
        result = {}
        prev = 0.0
        for name, t in self._laps:
            result[name] = t - prev
            prev = t
        return result