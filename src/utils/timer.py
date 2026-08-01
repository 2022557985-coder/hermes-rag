"""Timing utilities for performance measurement."""

import functools
import logging
import statistics
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("hermes_rag")


@contextmanager
def timer(name: str = "", log: object | None = None):
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
        self._start: float = time.perf_counter()
        self._laps: list[tuple] = []

    def lap(self, name: str) -> float:
        """Record a lap time."""
        elapsed = time.perf_counter() - self._start
        self._laps.append((name, elapsed))
        return elapsed

    def elapsed(self) -> float:
        """Return total elapsed time since start or last reset, in seconds."""
        return time.perf_counter() - self._start

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

    def durations(self) -> dict[str, float]:
        """Return a dict of lap name -> delta time (seconds).

        Returns:
            dict mapping lap names to their individual durations.
        """
        result: dict[str, float] = {}
        prev = 0.0
        for name, t in self._laps:
            result[name] = t - prev
            prev = t
        return result

    def to_dict(self) -> dict[str, Any]:
        """Return laps as a comprehensive dict with durations and cumulative times.

        Returns:
            Dict with 'laps' (list of {name, duration, cumulative}) and 'total'.
        """
        laps_data: list[dict[str, Any]] = []
        prev = 0.0
        total = 0.0
        for name, t in self._laps:
            duration = t - prev
            laps_data.append({
                "name": name,
                "duration": duration,
                "cumulative": t,
            })
            prev = t
            total = t
        return {
            "laps": laps_data,
            "total": total,
        }

    def mean(self) -> float:
        """Compute the mean duration of all laps.

        Returns:
            Mean lap duration in seconds. Returns 0.0 if no laps.
        """
        durations = list(self.durations().values())
        if not durations:
            return 0.0
        return sum(durations) / len(durations)

    @staticmethod
    def percentile(durations: list[float], p: float) -> float:
        """Compute the p-th percentile from a list of durations.

        Uses linear interpolation between adjacent values.

        Args:
            durations: List of timing durations in seconds.
            p: Percentile to compute (0.0 to 100.0).

        Returns:
            The p-th percentile value. Returns 0.0 if durations is empty.
        """
        if not durations:
            return 0.0
        if p < 0.0 or p > 100.0:
            raise ValueError("Percentile must be between 0.0 and 100.0")

        sorted_durations = sorted(durations)
        n = len(sorted_durations)

        if n == 1:
            return sorted_durations[0]

        rank = (p / 100.0) * (n - 1)
        lower = int(rank)
        upper = lower + 1 if lower < n - 1 else lower
        weight = rank - lower

        return sorted_durations[lower] * (1 - weight) + sorted_durations[upper] * weight


class TimerStats:
    """Accumulate multiple timing runs and compute statistics.

    Usage:
        stats = TimerStats()
        with stats:
            do_work()
        with stats:
            do_other_work()
        print(stats.summary())
    """

    def __init__(self, name: str = ""):
        self.name: str = name
        self._durations: list[float] = []
        self._start: float | None = None
        self._count: int = 0
        self._total: float = 0.0
        self._min: float | None = None
        self._max: float | None = None

    def __enter__(self) -> "TimerStats":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._start is not None:
            elapsed = time.perf_counter() - self._start
            self._durations.append(elapsed)
            self._count += 1
            self._total += elapsed
            if self._min is None or elapsed < self._min:
                self._min = elapsed
            if self._max is None or elapsed > self._max:
                self._max = elapsed
            self._start = None

    def record(self, duration: float) -> None:
        """Manually record a duration.

        Args:
            duration: Duration in seconds to record.
        """
        self._durations.append(duration)
        self._count += 1
        self._total += duration
        if self._min is None or duration < self._min:
            self._min = duration
        if self._max is None or duration > self._max:
            self._max = duration

    @property
    def count(self) -> int:
        """Number of recorded timing runs."""
        return self._count

    @property
    def total(self) -> float:
        """Total accumulated time."""
        return self._total

    def mean(self) -> float:
        """Mean duration of all runs."""
        if self._count == 0:
            return 0.0
        return self._total / self._count

    def median(self) -> float:
        """Median duration of all runs."""
        if not self._durations:
            return 0.0
        return float(statistics.median(self._durations))

    def stdev(self) -> float:
        """Standard deviation of all runs."""
        if len(self._durations) < 2:
            return 0.0
        return float(statistics.stdev(self._durations))

    def percentile(self, p: float) -> float:
        """Compute the p-th percentile of all recorded durations.

        Args:
            p: Percentile to compute (0.0 to 100.0).

        Returns:
            The p-th percentile duration.
        """
        return Stopwatch.percentile(self._durations, p)

    def summary(self) -> str:
        """Return a human-readable summary of timing statistics."""
        prefix = f"[{self.name}]" if self.name else ""
        return (
            f"{prefix} count={self._count}, "
            f"total={self._total:.4f}s, "
            f"mean={self.mean():.4f}s, "
            f"median={self.median():.4f}s, "
            f"stdev={self.stdev():.4f}s, "
            f"min={self._min:.4f}s, "
            f"max={self._max:.4f}s"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return timing statistics as a dict.

        Returns:
            Dict with count, total, mean, median, stdev, min, max, p50, p95, p99.
        """
        return {
            "name": self.name,
            "count": self._count,
            "total": round(self._total, 6),
            "mean": round(self.mean(), 6),
            "median": round(self.median(), 6),
            "stdev": round(self.stdev(), 6),
            "min": round(self._min, 6) if self._min is not None else 0.0,
            "max": round(self._max, 6) if self._max is not None else 0.0,
            "p50": round(self.percentile(50.0), 6),
            "p95": round(self.percentile(95.0), 6),
            "p99": round(self.percentile(99.0), 6),
        }

    def reset(self) -> None:
        """Reset all accumulated timing data."""
        self._durations.clear()
        self._start = None
        self._count = 0
        self._total = 0.0
        self._min = None
        self._max = None