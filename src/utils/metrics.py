"""Production metrics collection for Hermes-RAG.

Tracks cache hit rate, query latency percentiles, recall path distribution,
and system health indicators for production monitoring.
"""

import threading
import time
from collections import deque
from typing import Dict, Any, List, Optional


class MetricsCollector:
    """Thread-safe metrics collector for production monitoring.

    Tracks:
    - Cache hit rate (overall and per-window)
    - Query latency (avg, p50, p95, p99)
    - Recall path distribution (dense_only, sparse_only, both, cached)
    - Reranker usage and timeout rate
    - System memory and health
    """

    def __init__(self, window_size: int = 1000, latency_window: int = 500):
        self._lock = threading.Lock()

        # Counters
        self._total_queries: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._reranker_timeouts: int = 0
        self._reranker_used: int = 0
        self._dense_only_recalls: int = 0
        self._sparse_only_recalls: int = 0
        self._both_recalls: int = 0
        self._failed_queries: int = 0

        # Latency tracking (sliding window)
        self._latency_window = deque(maxlen=latency_window)

        # Per-component timing
        self._component_timings: Dict[str, deque] = {
            "cache_lookup": deque(maxlen=window_size),
            "query_expansion": deque(maxlen=window_size),
            "dense_retrieval": deque(maxlen=window_size),
            "sparse_retrieval": deque(maxlen=window_size),
            "rrf_fusion": deque(maxlen=window_size),
            "reranking": deque(maxlen=window_size),
            "total": deque(maxlen=window_size),
        }

        # Start time
        self._start_time = time.time()

    # ---- Recording methods ----

    def record_query(
        self,
        cached: bool = False,
        recall_paths: Optional[List[str]] = None,
        total_latency: float = 0.0,
        component_timings: Optional[Dict[str, float]] = None,
        reranker_used: bool = False,
        reranker_timed_out: bool = False,
    ) -> None:
        """Record a completed query."""
        with self._lock:
            self._total_queries += 1
            if cached:
                self._cache_hits += 1
            else:
                self._cache_misses += 1

            if reranker_used:
                self._reranker_used += 1
            if reranker_timed_out:
                self._reranker_timeouts += 1

            # Recall path distribution
            if recall_paths:
                if "dense" in recall_paths and "sparse" in recall_paths:
                    self._both_recalls += 1
                elif "dense" in recall_paths:
                    self._dense_only_recalls += 1
                elif "sparse" in recall_paths:
                    self._sparse_only_recalls += 1

            # Latency
            self._latency_window.append(total_latency)

            # Component timings
            if component_timings:
                for comp, duration in component_timings.items():
                    if comp in self._component_timings:
                        self._component_timings[comp].append(duration)

    def record_failure(self) -> None:
        """Record a failed query."""
        with self._lock:
            self._failed_queries += 1

    # ---- Query methods ----

    def get_cache_hit_rate(self) -> float:
        """Get current cache hit rate (0.0 to 1.0)."""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total

    def get_latency_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles: avg, p50, p95, p99."""
        with self._lock:
            if not self._latency_window:
                return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
            latencies = sorted(self._latency_window)
            n = len(latencies)
            return {
                "avg": sum(latencies) / n,
                "p50": latencies[int(n * 0.50)] if n > 0 else 0.0,
                "p95": latencies[int(n * 0.95)] if n > 1 else latencies[0],
                "p99": latencies[int(n * 0.99)] if n > 1 else latencies[0],
            }

    def get_recall_path_distribution(self) -> Dict[str, int]:
        """Get recall path distribution."""
        with self._lock:
            return {
                "dense_only": self._dense_only_recalls,
                "sparse_only": self._sparse_only_recalls,
                "both": self._both_recalls,
                "cached": self._cache_hits,
            }

    def get_component_avg_latency(self) -> Dict[str, float]:
        """Get average latency per component."""
        result = {}
        for comp, dq in self._component_timings.items():
            if dq:
                result[comp] = sum(dq) / len(dq)
            else:
                result[comp] = 0.0
        return result

    def get_full_report(self) -> Dict[str, Any]:
        """Get a comprehensive metrics report."""
        latencies = self.get_latency_percentiles()
        with self._lock:
            uptime = time.time() - self._start_time
            total = self._total_queries

        return {
            "uptime_seconds": round(uptime, 1),
            "total_queries": total,
            "qps": round(total / max(uptime, 1), 2),
            "cache": {
                "hit_rate": round(self.get_cache_hit_rate(), 4),
                "hits": self._cache_hits,
                "misses": self._cache_misses,
            },
            "latency": {
                "avg_ms": round(latencies["avg"] * 1000, 1),
                "p50_ms": round(latencies["p50"] * 1000, 1),
                "p95_ms": round(latencies["p95"] * 1000, 1),
                "p99_ms": round(latencies["p99"] * 1000, 1),
            },
            "recall_paths": self.get_recall_path_distribution(),
            "component_latency_ms": {
                k: round(v * 1000, 1)
                for k, v in self.get_component_avg_latency().items()
            },
            "reranker": {
                "usage_count": self._reranker_used,
                "timeout_count": self._reranker_timeouts,
                "timeout_rate": round(
                    self._reranker_timeouts / max(self._reranker_used, 1), 4
                ),
            },
            "failures": self._failed_queries,
            "failure_rate": round(
                self._failed_queries / max(total, 1), 4
            ),
        }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._total_queries = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._reranker_timeouts = 0
            self._reranker_used = 0
            self._dense_only_recalls = 0
            self._sparse_only_recalls = 0
            self._both_recalls = 0
            self._failed_queries = 0
            self._latency_window.clear()
            for dq in self._component_timings.values():
                dq.clear()
            self._start_time = time.time()


# Global metrics instance (thread-safe lazy init)
_metrics: Optional[MetricsCollector] = None
_metrics_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """Get or create the global metrics collector."""
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = MetricsCollector()
    return _metrics


def reset_metrics() -> None:
    """Reset the global metrics collector."""
    global _metrics
    _metrics = None