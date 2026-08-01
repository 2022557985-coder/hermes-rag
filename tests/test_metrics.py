"""Comprehensive tests for MetricsCollector."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.metrics import MetricsCollector, get_metrics, reset_metrics


class TestMetricsCollector:
    """Test metrics collection accuracy and thread safety."""

    def test_initial_state(self):
        m = MetricsCollector()
        assert m.get_cache_hit_rate() == 0.0
        latencies = m.get_latency_percentiles()
        assert latencies["avg"] == 0.0

    def test_record_cache_hit(self):
        m = MetricsCollector()
        m.record_query(cached=True, total_latency=0.01)
        assert m.get_cache_hit_rate() == 1.0

    def test_record_cache_miss(self):
        m = MetricsCollector()
        m.record_query(cached=False, total_latency=0.05)
        assert m.get_cache_hit_rate() == 0.0

    def test_cache_hit_rate_mixed(self):
        m = MetricsCollector()
        for _ in range(3):
            m.record_query(cached=True, total_latency=0.01)
        for _ in range(7):
            m.record_query(cached=False, total_latency=0.05)
        assert m.get_cache_hit_rate() == 0.3

    def test_latency_percentiles(self):
        m = MetricsCollector()
        for i in range(100):
            m.record_query(total_latency=float(i) / 100.0)
        latencies = m.get_latency_percentiles()
        assert latencies["p50"] > 0
        assert latencies["p95"] > latencies["p50"]
        assert latencies["p99"] >= latencies["p95"]

    def test_latency_empty(self):
        m = MetricsCollector()
        latencies = m.get_latency_percentiles()
        assert latencies["avg"] == 0.0
        assert latencies["p50"] == 0.0

    def test_recall_path_distribution(self):
        m = MetricsCollector()
        m.record_query(recall_paths=["dense"])
        m.record_query(recall_paths=["sparse"])
        m.record_query(recall_paths=["dense", "sparse"])
        m.record_query(cached=True, recall_paths=["cached"])
        dist = m.get_recall_path_distribution()
        assert dist["dense_only"] == 1
        assert dist["sparse_only"] == 1
        assert dist["both"] == 1
        assert dist["cached"] == 1

    def test_component_timings(self):
        m = MetricsCollector()
        m.record_query(
            component_timings={
                "cache_lookup": 0.001,
                "query_expansion": 0.01,
                "dense_retrieval": 0.05,
                "total": 0.1,
            }
        )
        avg = m.get_component_avg_latency()
        assert avg["cache_lookup"] > 0
        assert avg["query_expansion"] > 0

    def test_reranker_tracking(self):
        m = MetricsCollector()
        m.record_query(reranker_used=True)
        m.record_query(reranker_used=True, reranker_timed_out=True)
        report = m.get_full_report()
        assert report["reranker"]["usage_count"] == 2
        assert report["reranker"]["timeout_count"] == 1
        assert report["reranker"]["timeout_rate"] == 0.5

    def test_failure_tracking(self):
        m = MetricsCollector()
        m.record_query()
        m.record_failure()
        report = m.get_full_report()
        assert report["failures"] == 1
        # failure_rate = failures / total_queries = 1/1 = 1.0
        assert report["failure_rate"] == 1.0

    def test_qps_calculation(self):
        m = MetricsCollector()
        m.record_query()
        time.sleep(0.01)
        report = m.get_full_report()
        assert report["qps"] > 0

    def test_reset(self):
        m = MetricsCollector()
        m.record_query(cached=True, total_latency=0.1)
        m.reset()
        assert m.get_cache_hit_rate() == 0.0
        assert m._total_queries == 0

    def test_full_report_structure(self):
        m = MetricsCollector()
        m.record_query()
        report = m.get_full_report()
        required_keys = [
            "uptime_seconds", "total_queries", "qps",
            "cache", "latency", "recall_paths",
            "component_latency_ms", "reranker", "failures", "failure_rate",
        ]
        for key in required_keys:
            assert key in report, f"Missing key: {key}"

    def test_thread_safety(self):
        import threading
        m = MetricsCollector()
        errors = []

        def record_batch():
            try:
                for _ in range(100):
                    m.record_query(total_latency=0.01)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_batch) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert m._total_queries == 1000


class TestGlobalMetrics:
    """Test global metrics singleton."""

    def test_singleton(self):
        reset_metrics()
        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2

    def test_reset_creates_new(self):
        reset_metrics()
        m1 = get_metrics()
        m1.record_query()
        reset_metrics()
        m2 = get_metrics()
        assert m2._total_queries == 0