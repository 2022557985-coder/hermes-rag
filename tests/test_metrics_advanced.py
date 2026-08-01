"""Advanced tests for MetricsCollector: error rates, throughput, slow queries, health, Prometheus, memory."""

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from utils.metrics import MetricsCollector, get_metrics, reset_metrics
except ImportError:
    MetricsCollector = None
    get_metrics = None
    reset_metrics = None


class TestErrorRate:
    """Test get_error_rate and related methods."""

    def test_error_rate_zero_initially(self):
        """Error rate should be 0.0 initially."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        assert m.get_error_rate() == 0.0

    def test_error_rate_with_errors(self):
        """Error rate should reflect recorded errors."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        m.record_query()
        m.record_error("timeout")
        rate = m.get_error_rate()
        assert rate == 1.0, f"Expected error rate 1.0, got {rate}"

    def test_error_rate_mixed(self):
        """Mixed errors and queries should produce correct rate."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        for _ in range(5):
            m.record_query()
        m.record_error("timeout")
        rate = m.get_error_rate()
        assert rate == 0.2, f"Expected error rate 0.2, got {rate}"


class TestQueryThroughput:
    """Test get_query_throughput."""

    def test_throughput_initial(self):
        """Initial throughput should be 0.0."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        throughput = m.get_query_throughput()
        assert throughput == 0.0

    def test_throughput_nonzero(self):
        """Throughput should be non-zero after recording queries."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        m.record_query()
        time.sleep(0.01)
        throughput = m.get_query_throughput(window_seconds=60.0)
        assert throughput > 0.0, f"Expected throughput > 0, got {throughput}"

    def test_throughput_custom_window(self):
        """Throughput with custom window should work."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        for _ in range(10):
            m.record_query()
        throughput = m.get_query_throughput(window_seconds=3600.0)
        assert throughput > 0.0


class TestSlowQueries:
    """Test get_slow_queries."""

    def test_slow_queries_empty(self):
        """Empty slow queries should return empty list."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        result = m.get_slow_queries()
        assert result == []

    def test_slow_queries_recorded(self):
        """Recorded slow queries should be returned."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        m.record_query(query_text="slow query 1", total_latency=5.0)
        m.record_query(query_text="fast query", total_latency=0.1)
        result = m.get_slow_queries(top_n=10)
        assert len(result) == 2
        assert result[0][0] == "slow query 1"
        assert result[0][1] == 5.0

    def test_slow_queries_top_n(self):
        """Slow queries should respect top_n parameter."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        for i in range(10):
            m.record_query(query_text=f"query {i}", total_latency=float(i))
        result = m.get_slow_queries(top_n=3)
        assert len(result) == 3
        assert result[0][1] == 9.0  # Slowest first

    def test_slow_queries_sorted(self):
        """Slow queries should be sorted by latency descending."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        m.record_query(query_text="q1", total_latency=1.0)
        m.record_query(query_text="q2", total_latency=3.0)
        m.record_query(query_text="q3", total_latency=2.0)
        result = m.get_slow_queries()
        assert result[0][1] >= result[1][1] >= result[2][1], (
            f"Slow queries not sorted: {result}"
        )


class TestRecordError:
    """Test record_error with different error types."""

    def test_record_error_types(self):
        """Different error types should be tracked separately."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        m.record_error("timeout")
        m.record_error("network")
        m.record_error("timeout")
        m.record_error("oom")
        dist = m.get_error_distribution()
        assert dist["timeout"] == 2
        assert dist["network"] == 1
        assert dist["oom"] == 1

    def test_record_error_unknown(self):
        """Error without type should default to 'unknown'."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        m.record_error()
        dist = m.get_error_distribution()
        assert dist["unknown"] == 1


class TestErrorDistribution:
    """Test get_error_distribution."""

    def test_empty_distribution(self):
        """Empty error distribution should return empty dict."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        dist = m.get_error_distribution()
        assert dist == {}

    def test_distribution_is_copy(self):
        """Error distribution should return a copy."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        m.record_error("timeout")
        dist = m.get_error_distribution()
        dist["timeout"] = 999  # Modify the copy
        # Original should be unchanged
        original = m.get_error_distribution()
        assert original["timeout"] == 1


class TestHealthStatus:
    """Test get_health_status (GREEN, YELLOW, RED)."""

    def test_health_green_initially(self):
        """Initial health status should be GREEN."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        assert m.get_health_status() == "GREEN"

    def test_health_red_high_error_rate(self):
        """High error rate should trigger RED status."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        m.record_query()
        for _ in range(20):
            m.record_error("timeout")
        # Error rate = 20/1 > 0.1 -> RED
        assert m.get_health_status() == "RED"

    def test_health_red_high_failure_rate(self):
        """High failure rate should trigger RED status."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        for _ in range(10):
            m.record_query()
        for _ in range(10):
            m.record_failure()
        # Failure rate = 10/10 = 1.0 > 0.05 -> RED
        assert m.get_health_status() == "RED"

    def test_health_yellow_moderate_error_rate(self):
        """Moderate error rate should trigger YELLOW status."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        for _ in range(100):
            m.record_query()
        for _ in range(3):
            m.record_error("timeout")
        # Error rate = 3/100 = 0.03 > 0.02 -> YELLOW
        assert m.get_health_status() == "YELLOW"

    def test_health_yellow_high_latency(self):
        """High latency should trigger YELLOW status."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        for _ in range(100):
            m.record_query(total_latency=20.0)
        # p95 = 20.0 > 10.0 -> YELLOW
        status = m.get_health_status()
        assert status in ("YELLOW", "RED"), f"Expected YELLOW or RED, got {status}"


class TestPrometheusFormat:
    """Test to_prometheus_format."""

    def test_prometheus_format_keys(self):
        """Prometheus output should contain expected metrics."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        m.record_query()
        m.record_error("timeout")
        output = m.to_prometheus_format()
        expected_metrics = [
            "hermes_rag_total_queries",
            "hermes_rag_cache_hits",
            "hermes_rag_cache_misses",
            "hermes_rag_cache_hit_rate",
            "hermes_rag_failed_queries",
            "hermes_rag_error_rate",
            "hermes_rag_health_status",
            "hermes_rag_memory_rss_mb",
            "hermes_rag_uptime_seconds",
        ]
        for metric in expected_metrics:
            assert metric in output, f"Missing metric '{metric}' in Prometheus output"

    def test_prometheus_format_has_help(self):
        """Prometheus output should contain HELP lines."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        output = m.to_prometheus_format()
        assert "# HELP" in output
        assert "# TYPE" in output

    def test_prometheus_format_error_distribution(self):
        """Prometheus output should include error distribution."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        m.record_error("timeout")
        output = m.to_prometheus_format()
        assert "hermes_rag_errors_timeout" in output


class TestMemoryUsage:
    """Test get_memory_usage."""

    def test_memory_usage_has_rss(self):
        """Memory usage should include rss_mb."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        mem = m.get_memory_usage()
        assert "rss_mb" in mem, f"Missing 'rss_mb' in memory: {mem}"

    def test_memory_usage_rss_is_number(self):
        """rss_mb should be a numeric value."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        mem = m.get_memory_usage()
        assert isinstance(mem["rss_mb"], (int, float))


class TestConcurrentAccess:
    """Test concurrent access (multiple threads recording)."""

    def test_concurrent_error_recording(self):
        """Multiple threads recording errors should not corrupt data."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        errors = []

        def record_errors():
            try:
                for _ in range(50):
                    m.record_query()
                    m.record_error("timeout")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record_errors) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent recording: {errors}"
        assert m._total_queries == 200
        dist = m.get_error_distribution()
        assert dist.get("timeout", 0) == 200

    def test_concurrent_mixed_operations(self):
        """Multiple threads performing mixed operations should be safe."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        errors = []

        def mixed_ops():
            try:
                for i in range(25):
                    m.record_query(cached=i % 2 == 0, total_latency=0.01 * i)
                    m.record_query(recall_paths=["dense", "sparse"])
                    m.record_error(f"type_{i % 5}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mixed_ops) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent mixed ops: {errors}"
        assert m._total_queries == 200  # 25 * 2 * 4


class TestResetMetrics:
    """Test reset_metrics preserves collector reference."""

    def test_reset_preserves_reference(self):
        """Reset should preserve the same collector instance."""
        if reset_metrics is None or get_metrics is None:
            pytest.skip("Metrics module not available")
        reset_metrics()
        m1 = get_metrics()
        m1.record_query()
        reset_metrics()
        m2 = get_metrics()
        assert m1 is m2, "Reset should preserve the same collector instance"
        assert m2._total_queries == 0, "Reset should clear query count"

    def test_reset_clears_all_counters(self):
        """Reset should clear all counters."""
        if MetricsCollector is None:
            pytest.skip("MetricsCollector module not available")
        m = MetricsCollector()
        m.record_query(cached=True)
        m.record_error("timeout")
        m.record_failure()
        m.reset()
        assert m._total_queries == 0
        assert m._cache_hits == 0
        assert m._failed_queries == 0
        assert m.get_error_distribution() == {}

    def test_reset_creates_default_if_none(self):
        """Calling reset_metrics when _metrics is None should create default."""
        if reset_metrics is None or get_metrics is None:
            pytest.skip("Metrics module not available")
        # Force reset from a known state
        reset_metrics()
        m = get_metrics()
        assert m is not None
        assert m._total_queries == 0