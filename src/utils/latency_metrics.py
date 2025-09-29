"""
Latency Metrics Measurement and Analysis (GREEN phase)
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import contextmanager


class LatencyTimer:
    """Context manager for measuring execution time"""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed_ms = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000  # Convert to ms


class LatencyTracker:
    """Track multiple latency measurements"""

    def __init__(self):
        self.measurements: List[float] = []

    @contextmanager
    def time(self):
        """Context manager to measure and record latency"""
        timer = LatencyTimer()
        with timer:
            yield timer
        self.measurements.append(timer.elapsed_ms)

    def mean(self) -> float:
        """Calculate mean latency"""
        return np.mean(self.measurements)

    def std(self) -> float:
        """Calculate standard deviation"""
        return np.std(self.measurements)

    def percentiles(self) -> Dict[str, float]:
        """Calculate key percentiles"""
        return calculate_percentiles(np.array(self.measurements))


def calculate_percentiles(latencies: np.ndarray) -> Dict[str, float]:
    """
    Calculate latency percentiles

    Args:
        latencies: Array of latency measurements (in milliseconds)

    Returns:
        Dict with p50, p90, p95, p99 values
    """
    return {
        "p50": np.percentile(latencies, 50),
        "p90": np.percentile(latencies, 90),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99)
    }


def calculate_latency_stats(latencies: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive latency statistics

    Args:
        latencies: Array of latency measurements (in milliseconds)

    Returns:
        Dict with mean, std, min, max, percentiles
    """
    percentiles = calculate_percentiles(latencies)

    return {
        "mean": np.mean(latencies),
        "std": np.std(latencies),
        "min": np.min(latencies),
        "max": np.max(latencies),
        **percentiles
    }


def plot_latency_histogram(
    latencies: np.ndarray,
    output_path: Path,
    title: str = "Inference Latency Distribution"
) -> None:
    """
    Create latency histogram with percentile markers

    Args:
        latencies: Array of latency measurements (in milliseconds)
        output_path: Path to save plot
        title: Plot title
    """
    percentiles = calculate_percentiles(latencies)
    stats = calculate_latency_stats(latencies)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Histogram
    n, bins, patches = ax.hist(latencies, bins=50, alpha=0.7, color='skyblue', edgecolor='black')

    # Add percentile lines
    colors = {'p50': 'green', 'p90': 'orange', 'p95': 'red', 'p99': 'darkred'}
    for pct, color in colors.items():
        ax.axvline(
            percentiles[pct],
            color=color,
            linestyle='--',
            linewidth=2,
            label=f'{pct.upper()}: {percentiles[pct]:.2f}ms'
        )

    # Add mean line
    ax.axvline(
        stats['mean'],
        color='blue',
        linestyle='-',
        linewidth=2,
        label=f'Mean: {stats["mean"]:.2f}ms'
    )

    ax.set_xlabel('Latency (milliseconds)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add statistics text box
    stats_text = f"""
    Mean: {stats['mean']:.2f} ms
    Std: {stats['std']:.2f} ms
    Min: {stats['min']:.2f} ms
    Max: {stats['max']:.2f} ms
    P50: {percentiles['p50']:.2f} ms
    P90: {percentiles['p90']:.2f} ms
    P95: {percentiles['p95']:.2f} ms
    P99: {percentiles['p99']:.2f} ms
    """

    ax.text(
        0.98, 0.97,
        stats_text.strip(),
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Latency histogram saved to: {output_path}")


def format_latency_table(stats: Dict[str, float]) -> str:
    """
    Format latency statistics as markdown table

    Args:
        stats: Statistics dictionary from calculate_latency_stats()

    Returns:
        Markdown-formatted table string
    """
    table = """
| Metric | Value (ms) |
|--------|-----------|
| Mean   | {mean:.2f} |
| Std Dev| {std:.2f}  |
| Min    | {min:.2f}  |
| Max    | {max:.2f}  |
| P50    | {p50:.2f}  |
| P90    | {p90:.2f}  |
| P95    | {p95:.2f}  |
| P99    | {p99:.2f}  |
""".format(**stats)

    return table.strip()