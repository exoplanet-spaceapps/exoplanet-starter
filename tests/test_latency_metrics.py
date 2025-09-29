"""
Tests for Latency Metrics (TDD RED phase)
"""
# UTF-8 fix
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pytest
import numpy as np
import time
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


class TestLatencyMetrics:
    """Test suite for latency measurement and percentiles"""

    @pytest.fixture
    def sample_latencies(self):
        """Create sample latency measurements"""
        np.random.seed(42)
        # Simulate 1000 inference calls with realistic latencies
        latencies = np.random.lognormal(mean=np.log(0.05), sigma=0.3, size=1000)
        return latencies

    def test_latency_percentiles_calculation(self, sample_latencies):
        """Test: Should calculate 50th, 90th, 95th, 99th percentiles"""
        from utils.latency_metrics import calculate_percentiles

        percentiles = calculate_percentiles(sample_latencies)

        assert "p50" in percentiles
        assert "p90" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles

        # Percentiles should be in ascending order
        assert percentiles["p50"] <= percentiles["p90"]
        assert percentiles["p90"] <= percentiles["p95"]
        assert percentiles["p95"] <= percentiles["p99"]

    def test_latency_context_manager(self):
        """Test: Should measure execution time with context manager"""
        from utils.latency_metrics import LatencyTimer

        timer = LatencyTimer()

        with timer:
            time.sleep(0.01)  # Simulate 10ms operation

        assert timer.elapsed_ms >= 10.0
        assert timer.elapsed_ms < 50.0  # Should not be too slow

    def test_multiple_timing_measurements(self):
        """Test: Should collect multiple timing measurements"""
        from utils.latency_metrics import LatencyTracker

        tracker = LatencyTracker()

        for _ in range(10):
            with tracker.time():
                time.sleep(0.005)  # 5ms each

        assert len(tracker.measurements) == 10
        assert tracker.mean() >= 5.0

    def test_latency_summary_statistics(self, sample_latencies):
        """Test: Should calculate comprehensive statistics"""
        from utils.latency_metrics import calculate_latency_stats

        stats = calculate_latency_stats(sample_latencies)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "p50" in stats
        assert "p90" in stats
        assert "p95" in stats
        assert "p99" in stats

    def test_latency_histogram_creation(self, sample_latencies, tmp_path):
        """Test: Should create latency histogram with percentile markers"""
        from utils.latency_metrics import plot_latency_histogram

        output_path = tmp_path / "latency_histogram.png"

        plot_latency_histogram(sample_latencies, output_path)

        assert output_path.exists()


if __name__ == "__main__":
    print("🧪 Running Latency Metrics Tests (TDD RED Phase)")
    print("="*60)
    pytest.main([__file__, '-v'])