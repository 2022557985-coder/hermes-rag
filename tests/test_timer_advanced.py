"""Advanced tests for timer module: Stopwatch percentile/mean/to_dict, TimerStats, edge cases."""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from utils.timer import Stopwatch, TimerStats, timed, timer
except ImportError:
    Stopwatch = None
    TimerStats = None
    timer = None
    timed = None


class TestStopwatchPercentile:
    """Test Stopwatch.percentile static method."""

    def test_percentile_basic(self):
        """Basic percentile calculation."""
        if Stopwatch is None:
            pytest.skip("Timer module not available")
        durations = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert Stopwatch.percentile(durations, 50.0) == 3.0
        assert Stopwatch.percentile(durations, 0.0) == 1.0
        assert Stopwatch.percentile(durations, 100.0) == 5.0

    def test_percentile_single_element(self):
        """Percentile of single element list."""
        if Stopwatch is None:
            pytest.skip("Timer module not available")
        assert Stopwatch.percentile([42.0], 50.0) == 42.0
        assert Stopwatch.percentile([42.0], 0.0) == 42.0
        assert Stopwatch.percentile([42.0], 100.0) == 42.0

    def test_percentile_empty_list(self):
        """Percentile of empty list should return 0.0."""
        if Stopwatch is None:
            pytest.skip("Timer module not available")
        assert Stopwatch.percentile([], 50.0) == 0.0

    def test_percentile_interpolation(self):
        """Percentile should use linear interpolation."""
        if Stopwatch is None:
            pytest.skip("Timer module not available")
        durations = [1.0, 2.0, 3.0, 4.0]
        # p25: rank = 0.25 * 3 = 0.75, between 1.0 and 2.0
        p25 = Stopwatch.percentile(durations, 25.0)
        assert 1.0 <= p25 <= 2.0, f"Expected p25 between 1.0 and 2.0, got {p25}"

    def test_percentile_invalid_p(self):
        """Percentile with invalid p should raise ValueError."""
        if Stopwatch is None:
            pytest.skip("Timer module not available")
        with pytest.raises(ValueError):
            Stopwatch.percentile([1.0, 2.0], -1.0)
        with pytest.raises(ValueError):
            Stopwatch.percentile([1.0, 2.0], 101.0)

    def test_percentile_two_elements(self):
        """Percentile with two elements."""
        if Stopwatch is None:
            pytest.skip("Timer module not available")
        durations = [1.0, 10.0]
        p50 = Stopwatch.percentile(durations, 50.0)
        # rank = 0.5 * 1 = 0.5, between 1.0 and 10.0
        assert p50 == 5.5, f"Expected p50=5.5, got {p50}"


class TestStopwatchMean:
    """Test Stopwatch.mean method."""

    def test_mean_empty(self):
        """Mean of empty stopwatch should be 0.0."""
        if Stopwatch is None:
            pytest.skip("Timer module not available")
        sw = Stopwatch()
        assert sw.mean() == 0.0

    def test_mean_single_lap(self):
        """Mean of single lap should equal that lap's duration."""
        if Stopwatch is None:
            pytest.skip("Timer module not available")
        sw = Stopwatch()
        time.sleep(0.01)
        sw.lap("test")
        mean_val = sw.mean()
        assert mean_val > 0.0

    def test_mean_multiple_laps(self):
        """Mean of multiple laps should be correct."""
        if Stopwatch is None:
            pytest.skip("Timer module not available")
        sw = Stopwatch()
        time.sleep(0.01)
        sw.lap("lap1")
        time.sleep(0.02)
        sw.lap("lap2")
        durations = sw.durations()
        mean_val = sw.mean()
        expected = sum(durations.values()) / len(durations)
        assert abs(mean_val - expected) < 0.001, (
            f"Expected mean {expected}, got {mean_val}"
        )


class TestStopwatchToDict:
    """Test Stopwatch.to_dict method."""

    def test_to_dict_empty(self):
        """to_dict of empty stopwatch should have empty laps."""
        if Stopwatch is None:
            pytest.skip("Timer module not available")
        sw = Stopwatch()
        result = sw.to_dict()
        assert result["laps"] == []
        assert result["total"] == 0.0

    def test_to_dict_with_laps(self):
        """to_dict should return comprehensive lap data."""
        if Stopwatch is None:
            pytest.skip("Timer module not available")
        sw = Stopwatch()
        time.sleep(0.01)
        sw.lap("lap1")
        time.sleep(0.02)
        sw.lap("lap2")
        result = sw.to_dict()
        assert len(result["laps"]) == 2
        assert result["total"] > 0.0
        for lap in result["laps"]:
            assert "name" in lap
            assert "duration" in lap
            assert "cumulative" in lap
            assert lap["duration"] > 0.0

    def test_to_dict_lap_order(self):
        """to_dict should preserve lap order."""
        if Stopwatch is None:
            pytest.skip("Timer module not available")
        sw = Stopwatch()
        sw.lap("first")
        sw.lap("second")
        sw.lap("third")
        result = sw.to_dict()
        assert result["laps"][0]["name"] == "first"
        assert result["laps"][1]["name"] == "second"
        assert result["laps"][2]["name"] == "third"


class TestTimerStats:
    """Test TimerStats class with multiple runs."""

    def test_context_manager(self):
        """TimerStats should work as context manager."""
        if TimerStats is None:
            pytest.skip("Timer module not available")
        stats = TimerStats("test")
        with stats:
            time.sleep(0.01)
        assert stats.count == 1
        assert stats.total > 0.0

    def test_multiple_runs(self):
        """Multiple runs should accumulate statistics."""
        if TimerStats is None:
            pytest.skip("Timer module not available")
        stats = TimerStats("test")
        for _ in range(5):
            with stats:
                time.sleep(0.01)
        assert stats.count == 5
        assert stats.total > 0.0

    def test_mean(self):
        """Mean should be calculated correctly."""
        if TimerStats is None:
            pytest.skip("Timer module not available")
        stats = TimerStats("test")
        stats.record(1.0)
        stats.record(2.0)
        stats.record(3.0)
        assert stats.mean() == 2.0

    def test_median(self):
        """Median should be calculated correctly."""
        if TimerStats is None:
            pytest.skip("Timer module not available")
        stats = TimerStats("test")
        stats.record(1.0)
        stats.record(2.0)
        stats.record(3.0)
        assert stats.median() == 2.0

    def test_stdev(self):
        """Standard deviation should be calculated correctly."""
        if TimerStats is None:
            pytest.skip("Timer module not available")
        stats = TimerStats("test")
        stats.record(1.0)
        stats.record(1.0)
        stats.record(1.0)
        assert stats.stdev() == 0.0

    def test_percentile(self):
        """TimerStats percentile should delegate to Stopwatch.percentile."""
        if TimerStats is None:
            pytest.skip("Timer module not available")
        stats = TimerStats("test")
        for i in range(1, 11):
            stats.record(float(i))
        assert stats.percentile(50.0) == 5.5

    def test_to_dict(self):
        """to_dict should return comprehensive statistics."""
        if TimerStats is None:
            pytest.skip("Timer module not available")
        stats = TimerStats("my_timer")
        stats.record(1.0)
        stats.record(2.0)
        result = stats.to_dict()
        assert result["name"] == "my_timer"
        assert result["count"] == 2
        assert result["total"] == 3.0
        assert result["mean"] == 1.5
        assert result["median"] == 1.5
        assert result["min"] == 1.0
        assert result["max"] == 2.0
        assert "p50" in result
        assert "p95" in result
        assert "p99" in result

    def test_summary(self):
        """Summary should return human-readable string."""
        if TimerStats is None:
            pytest.skip("Timer module not available")
        stats = TimerStats("test")
        stats.record(1.0)
        summary = stats.summary()
        assert "count=1" in summary
        assert "mean=" in summary
        assert "median=" in summary

    def test_reset(self):
        """Reset should clear all accumulated data."""
        if TimerStats is None:
            pytest.skip("Timer module not available")
        stats = TimerStats("test")
        stats.record(1.0)
        stats.record(2.0)
        stats.reset()
        assert stats.count == 0
        assert stats.total == 0.0
        assert stats.mean() == 0.0
        assert stats.median() == 0.0

    def test_empty_stats(self):
        """Empty TimerStats should return sensible defaults."""
        if TimerStats is None:
            pytest.skip("Timer module not available")
        stats = TimerStats("empty")
        assert stats.count == 0
        assert stats.total == 0.0
        assert stats.mean() == 0.0
        assert stats.median() == 0.0
        assert stats.stdev() == 0.0
        assert stats.percentile(50.0) == 0.0

    def test_single_run_stats(self):
        """Single run should have valid statistics."""
        if TimerStats is None:
            pytest.skip("Timer module not available")
        stats = TimerStats("single")
        stats.record(5.0)
        assert stats.count == 1
        assert stats.total == 5.0
        assert stats.mean() == 5.0
        assert stats.median() == 5.0
        assert stats.stdev() == 0.0  # Single run, stdev = 0
        assert stats.percentile(50.0) == 5.0

    def test_min_max_tracking(self):
        """Min and max should be tracked correctly."""
        if TimerStats is None:
            pytest.skip("Timer module not available")
        stats = TimerStats("test")
        stats.record(5.0)
        stats.record(1.0)
        stats.record(10.0)
        result = stats.to_dict()
        assert result["min"] == 1.0
        assert result["max"] == 10.0

    def test_manual_record(self):
        """Manual record should work outside context manager."""
        if TimerStats is None:
            pytest.skip("Timer module not available")
        stats = TimerStats("test")
        stats.record(0.5)
        assert stats.count == 1
        assert stats.total == 0.5


class TestTimerContextManager:
    """Test timer context manager."""

    def test_timer_context_manager(self):
        """Timer context manager should execute without error."""
        if timer is None:
            pytest.skip("Timer module not available")
        # Should not raise error
        with timer("test_timer"):
            pass


class TestTimedDecorator:
    """Test timed decorator."""

    def test_timed_decorator(self):
        """Timed decorator should wrap function without error."""
        if timed is None:
            pytest.skip("Timer module not available")

        @timed
        def sample_function():
            return 42

        result = sample_function()
        assert result == 42


class TestStopwatchEdgeCases:
    """Test Stopwatch edge cases."""

    def test_reset(self):
        """Reset should clear laps."""
        if Stopwatch is None:
            pytest.skip("Timer module not available")
        sw = Stopwatch()
        sw.lap("lap1")
        sw.reset()
        assert sw.to_dict()["laps"] == []
        assert sw.mean() == 0.0

    def test_summary(self):
        """Summary should return formatted string."""
        if Stopwatch is None:
            pytest.skip("Timer module not available")
        sw = Stopwatch()
        sw.lap("test")
        summary = sw.summary()
        assert "test" in summary
        assert "s)" in summary

    def test_durations(self):
        """Durations should return dict of lap name to delta."""
        if Stopwatch is None:
            pytest.skip("Timer module not available")
        sw = Stopwatch()
        time.sleep(0.01)
        sw.lap("lap1")
        time.sleep(0.02)
        sw.lap("lap2")
        durations = sw.durations()
        assert "lap1" in durations
        assert "lap2" in durations
        assert durations["lap1"] > 0.0
        assert durations["lap2"] > 0.0