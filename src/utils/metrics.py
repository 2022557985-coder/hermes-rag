"""Production metrics collection for Hermes-RAG.

Tracks cache hit rate, query latency percentiles, recall path distribution,
error rates, memory usage, and system health indicators for production monitoring.
"""

import os
import threading
import time
from collections import deque
from typing import Any

# Try to import psutil for memory tracking; fall back to os-based approach
try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


class MetricsCollector:
    """Thread-safe metrics collector for production monitoring.

    Tracks:
    - Cache hit rate (overall and per-window)
    - Query latency (avg, p50, p95, p99)
    - Recall path distribution (dense_only, sparse_only, both, cached)
    - Reranker usage and timeout rate
    - Error rates and error type distribution
    - Slow query tracking
    - System memory and health
    """

    def __init__(self, window_size: int = 1000, latency_window: int = 500):
        self._lock = threading.RLock()

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

        # Error tracking
        self._error_counts: dict[str, int] = {}

        # Latency tracking (sliding window)
        self._latency_window: deque = deque(maxlen=latency_window)

        # Query timestamp tracking for throughput calculation
        self._query_timestamps: deque = deque(maxlen=window_size)

        # Slow query tracking (stores (query_text, latency) tuples)
        self._slow_queries: deque = deque(maxlen=window_size)

        # Per-component timing
        self._component_timings: dict[str, deque] = {
            "cache_lookup": deque(maxlen=window_size),
            "query_expansion": deque(maxlen=window_size),
            "dense_retrieval": deque(maxlen=window_size),
            "sparse_retrieval": deque(maxlen=window_size),
            "rrf_fusion": deque(maxlen=window_size),
            "reranking": deque(maxlen=window_size),
            "total": deque(maxlen=window_size),
        }

        # Start time
        self._start_time: float = time.time()

    # ---- Recording methods ----

    def record_query(
        self,
        cached: bool = False,
        recall_paths: list[str] | None = None,
        total_latency: float = 0.0,
        component_timings: dict[str, float] | None = None,
        reranker_used: bool = False,
        reranker_timed_out: bool = False,
        query_text: str = "",
    ) -> None:
        """Record a completed query.

        Args:
            cached: Whether the result was served from cache.
            recall_paths: List of recall paths used (e.g., ['dense', 'sparse']).
            total_latency: Total query latency in seconds.
            component_timings: Dict of component name -> latency in seconds.
            reranker_used: Whether the reranker was used.
            reranker_timed_out: Whether the reranker timed out.
            query_text: The query text (for slow query tracking).
        """
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

            # Query timestamp for throughput
            self._query_timestamps.append(time.time())

            # Slow query tracking
            if query_text and total_latency > 0:
                self._slow_queries.append((query_text, total_latency))

            # Component timings
            if component_timings:
                for comp, duration in component_timings.items():
                    if comp in self._component_timings:
                        self._component_timings[comp].append(duration)

    def record_failure(self) -> None:
        """Record a failed query."""
        with self._lock:
            self._failed_queries += 1

    def record_error(self, error_type: str = "unknown") -> None:
        """Record an error with type categorization.

        Args:
            error_type: Category of the error (e.g., 'timeout', 'oom', 'network').
        """
        with self._lock:
            self._error_counts[error_type] = self._error_counts.get(error_type, 0) + 1

    # ---- Query methods ----

    def get_cache_hit_rate(self) -> float:
        """Get current cache hit rate (0.0 to 1.0)."""
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        return self._cache_hits / total

    def get_error_rate(self) -> float:
        """Get current error rate (0.0 to 1.0).

        Returns:
            Ratio of error events to total queries. Returns 0.0 if no queries.
        """
        with self._lock:
            total_errors = sum(self._error_counts.values())
            if self._total_queries == 0:
                return 0.0
            return total_errors / self._total_queries

    def get_error_distribution(self) -> dict[str, int]:
        """Get error type distribution.

        Returns:
            Dict mapping error type strings to their counts.
        """
        with self._lock:
            return dict(self._error_counts)

    def get_query_throughput(self, window_seconds: float = 60.0) -> float:
        """Get queries per second over the last N seconds.

        Args:
            window_seconds: Time window in seconds for throughput calculation.

        Returns:
            Queries per second as a float.
        """
        with self._lock:
            if not self._query_timestamps:
                return 0.0

            now = time.time()
            cutoff = now - window_seconds
            recent = [ts for ts in self._query_timestamps if ts >= cutoff]
            if not recent:
                return 0.0
            return len(recent) / min(window_seconds, now - recent[0])

    def get_slow_queries(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Get the top N slowest queries.

        Args:
            top_n: Number of slowest queries to return.

        Returns:
            List of (query_text, latency_seconds) tuples sorted slowest first.
        """
        with self._lock:
            if not self._slow_queries:
                return []
            sorted_queries = sorted(
                self._slow_queries, key=lambda x: x[1], reverse=True
            )
            return sorted_queries[:top_n]

    def get_latency_percentiles(self) -> dict[str, float]:
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

    def get_recall_path_distribution(self) -> dict[str, int]:
        """Get recall path distribution."""
        with self._lock:
            return {
                "dense_only": self._dense_only_recalls,
                "sparse_only": self._sparse_only_recalls,
                "both": self._both_recalls,
                "cached": self._cache_hits,
            }

    def get_component_avg_latency(self) -> dict[str, float]:
        """Get average latency per component."""
        result: dict[str, float] = {}
        for comp, dq in self._component_timings.items():
            if dq:
                result[comp] = sum(dq) / len(dq)
            else:
                result[comp] = 0.0
        return result

    def get_memory_usage(self) -> dict[str, float]:
        """Get current memory usage in MB.

        Uses psutil if available, otherwise falls back to basic OS-level info.

        Returns:
            Dict with 'rss_mb' (resident memory) and optionally 'vms_mb' and 'percent'.
        """
        if _PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())
                mem_info = process.memory_info()
                return {
                    "rss_mb": round(mem_info.rss / (1024 * 1024), 2),
                    "vms_mb": round(mem_info.vms / (1024 * 1024), 2),
                    "percent": round(process.memory_percent(), 2),
                }
            except Exception:
                pass

        # Fallback: basic info from os (not available on all platforms)
        try:
            # On some platforms, we can get basic usage via resource
            import resource

            usage = resource.getrusage(resource.RUSAGE_SELF)
            return {
                "rss_mb": round(usage.ru_maxrss / 1024, 2),
            }
        except (ImportError, AttributeError):
            return {"rss_mb": -1.0}

    def get_health_status(self) -> str:
        """Get system health status based on key metrics thresholds.

        Returns:
            'GREEN' - All metrics within healthy thresholds.
            'YELLOW' - Some metrics are degraded.
            'RED' - Critical metrics are failing.
        """
        with self._lock:
            # Check error rate
            total_errors = sum(self._error_counts.values())
            error_rate = total_errors / max(self._total_queries, 1)

            # Check failure rate
            failure_rate = self._failed_queries / max(self._total_queries, 1)

            # Check latency (inline to avoid deadlock with _lock)
            if not self._latency_window:
                latencies = {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
            else:
                latencies_sorted = sorted(self._latency_window)
                n = len(latencies_sorted)
                latencies = {
                    "avg": sum(latencies_sorted) / n,
                    "p50": latencies_sorted[int(n * 0.50)] if n > 0 else 0.0,
                    "p95": latencies_sorted[int(n * 0.95)] if n > 1 else latencies_sorted[0],
                    "p99": latencies_sorted[int(n * 0.99)] if n > 1 else latencies_sorted[0],
                }

            # RED conditions: critical failures
            if error_rate > 0.1 or failure_rate > 0.05:
                return "RED"
            if latencies.get("p95", 0) > 60.0:  # p95 > 60s
                return "RED"
            if self._reranker_timeouts > 0 and self._reranker_used > 0:
                timeout_rate = self._reranker_timeouts / self._reranker_used
                if timeout_rate > 0.5:
                    return "RED"

            # YELLOW conditions: degraded but not critical
            if error_rate > 0.02 or failure_rate > 0.01:
                return "YELLOW"
            if latencies.get("p95", 0) > 10.0:  # p95 > 10s
                return "YELLOW"
            if self._reranker_used > 0:
                timeout_rate = self._reranker_timeouts / self._reranker_used
                if timeout_rate > 0.1:
                    return "YELLOW"

            # GREEN: all healthy
            return "GREEN"

    def to_prometheus_format(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            Multi-line string in Prometheus exposition format.
        """
        with self._lock:
            lines: list[str] = []

            lines.append("# HELP hermes_rag_total_queries Total number of queries processed.")
            lines.append("# TYPE hermes_rag_total_queries counter")
            lines.append(f"hermes_rag_total_queries {self._total_queries}")

            lines.append("# HELP hermes_rag_cache_hits Total cache hits.")
            lines.append("# TYPE hermes_rag_cache_hits counter")
            lines.append(f"hermes_rag_cache_hits {self._cache_hits}")

            lines.append("# HELP hermes_rag_cache_misses Total cache misses.")
            lines.append("# TYPE hermes_rag_cache_misses counter")
            lines.append(f"hermes_rag_cache_misses {self._cache_misses}")

            lines.append("# HELP hermes_rag_cache_hit_rate Cache hit rate.")
            lines.append("# TYPE hermes_rag_cache_hit_rate gauge")
            lines.append(f"hermes_rag_cache_hit_rate {self.get_cache_hit_rate()}")

            lines.append("# HELP hermes_rag_failed_queries Total failed queries.")
            lines.append("# TYPE hermes_rag_failed_queries counter")
            lines.append(f"hermes_rag_failed_queries {self._failed_queries}")

            lines.append("# HELP hermes_rag_error_rate Error rate.")
            lines.append("# TYPE hermes_rag_error_rate gauge")
            lines.append(f"hermes_rag_error_rate {self.get_error_rate()}")

            # Error distribution
            for error_type, count in self._error_counts.items():
                safe_name = error_type.replace(" ", "_").replace("-", "_")
                lines.append(
                    f"# HELP hermes_rag_errors_{safe_name} Error count for type '{error_type}'."
                )
                lines.append(f"# TYPE hermes_rag_errors_{safe_name} counter")
                lines.append(f"hermes_rag_errors_{safe_name} {count}")

            lines.append("# HELP hermes_rag_reranker_used Total reranker usages.")
            lines.append("# TYPE hermes_rag_reranker_used counter")
            lines.append(f"hermes_rag_reranker_used {self._reranker_used}")

            lines.append("# HELP hermes_rag_reranker_timeouts Total reranker timeouts.")
            lines.append("# TYPE hermes_rag_reranker_timeouts counter")
            lines.append(f"hermes_rag_reranker_timeouts {self._reranker_timeouts}")

            # Latency percentiles
            latencies = self.get_latency_percentiles()
            for key, label in [("avg", "avg"), ("p50", "p50"), ("p95", "p95"), ("p99", "p99")]:
                lines.append(f"# HELP hermes_rag_latency_{label}_seconds Query latency {label}.")
                lines.append(f"# TYPE hermes_rag_latency_{label}_seconds gauge")
                lines.append(f"hermes_rag_latency_{label}_seconds {latencies.get(key, 0.0)}")

            # Recall path distribution
            recall = self.get_recall_path_distribution()
            for path, count in recall.items():
                lines.append(f"# HELP hermes_rag_recall_{path} Recall path '{path}' count.")
                lines.append(f"# TYPE hermes_rag_recall_{path} counter")
                lines.append(f"hermes_rag_recall_{path} {count}")

            # Component latencies
            comp_lat = self.get_component_avg_latency()
            for comp, lat in comp_lat.items():
                safe_comp = comp.replace(" ", "_")
                lines.append(f"# HELP hermes_rag_component_latency_{safe_comp}_seconds Component latency.")
                lines.append(f"# TYPE hermes_rag_component_latency_{safe_comp}_seconds gauge")
                lines.append(f"hermes_rag_component_latency_{safe_comp}_seconds {lat}")

            # Health status
            lines.append("# HELP hermes_rag_health_status Health status (0=RED, 1=YELLOW, 2=GREEN).")
            lines.append("# TYPE hermes_rag_health_status gauge")
            status_map = {"RED": 0, "YELLOW": 1, "GREEN": 2}
            lines.append(f"hermes_rag_health_status {status_map.get(self.get_health_status(), -1)}")

            # Memory
            mem = self.get_memory_usage()
            lines.append("# HELP hermes_rag_memory_rss_mb Resident memory in MB.")
            lines.append("# TYPE hermes_rag_memory_rss_mb gauge")
            lines.append(f"hermes_rag_memory_rss_mb {mem.get('rss_mb', -1)}")

            # Uptime
            lines.append("# HELP hermes_rag_uptime_seconds Process uptime in seconds.")
            lines.append("# TYPE hermes_rag_uptime_seconds gauge")
            lines.append(f"hermes_rag_uptime_seconds {time.time() - self._start_time}")

            return "\n".join(lines) + "\n"

    def get_full_report(self) -> dict[str, Any]:
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
            "errors": {
                "distribution": self.get_error_distribution(),
                "error_rate": round(self.get_error_rate(), 4),
            },
            "failures": self._failed_queries,
            "failure_rate": round(
                self._failed_queries / max(total, 1), 4
            ),
            "health": self.get_health_status(),
            "memory": self.get_memory_usage(),
            "throughput_qps": round(self.get_query_throughput(), 2),
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
            self._error_counts.clear()
            self._latency_window.clear()
            self._query_timestamps.clear()
            self._slow_queries.clear()
            for dq in self._component_timings.values():
                dq.clear()
            self._start_time = time.time()


# Global metrics instance (thread-safe lazy init)
_metrics: MetricsCollector | None = None
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
    """Reset the global metrics collector without losing the collector reference.

    This preserves the global instance so that any existing references to the
    collector remain valid.
    """
    global _metrics
    if _metrics is not None:
        _metrics.reset()
    else:
        _metrics = MetricsCollector()