"""
Latency metrics tracking and analysis for model inference performance
"""

import time
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Any
from pathlib import Path


class LatencyTracker:
    """Track inference latency for performance benchmarking"""

    def __init__(self):
        """Initialize latency tracker"""
        self.latencies: List[float] = []
        self.start_time: float = 0.0

    def start(self):
        """Start timing"""
        self.start_time = time.time()

    def stop(self):
        """Stop timing and record latency"""
        if self.start_time == 0.0:
            raise RuntimeError("Must call start() before stop()")
        latency = time.time() - self.start_time
        self.latencies.append(latency)
        self.start_time = 0.0
        return latency

    def reset(self):
        """Reset all recorded latencies"""
        self.latencies = []
        self.start_time = 0.0

    def get_latencies(self) -> np.ndarray:
        """Get all recorded latencies in milliseconds"""
        return np.array(self.latencies) * 1000  # Convert to ms

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.start_time != 0.0:
            self.stop()


def calculate_latency_stats(latencies: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive latency statistics including percentiles

    Args:
        latencies: Array of latency values in milliseconds

    Returns:
        Dictionary containing latency statistics
    """
    if len(latencies) == 0:
        return {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'p50': 0.0,
            'p90': 0.0,
            'p95': 0.0,
            'p99': 0.0
        }

    stats = {
        'mean': float(np.mean(latencies)),
        'median': float(np.median(latencies)),
        'std': float(np.std(latencies)),
        'min': float(np.min(latencies)),
        'max': float(np.max(latencies)),
        'p50': float(np.percentile(latencies, 50)),
        'p90': float(np.percentile(latencies, 90)),
        'p95': float(np.percentile(latencies, 95)),
        'p99': float(np.percentile(latencies, 99))
    }

    return stats


def plot_latency_histogram(
    latencies: np.ndarray,
    title: str = "Inference Latency Distribution",
    output_path: str = None
) -> go.Figure:
    """
    Create interactive histogram of latency distribution with percentile markers

    Args:
        latencies: Array of latency values in milliseconds
        title: Plot title
        output_path: Optional path to save HTML file

    Returns:
        Plotly figure object
    """
    stats = calculate_latency_stats(latencies)

    # Create histogram
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=latencies,
        nbinsx=50,
        name='Latency Distribution',
        marker_color='rgba(52, 152, 219, 0.7)',
        hovertemplate='Latency: %{x:.2f}ms<br>Count: %{y}<extra></extra>'
    ))

    # Add percentile lines
    percentiles = {
        'P50 (Median)': stats['p50'],
        'P90': stats['p90'],
        'P95': stats['p95'],
        'P99': stats['p99']
    }

    colors = {
        'P50 (Median)': 'green',
        'P90': 'orange',
        'P95': 'red',
        'P99': 'darkred'
    }

    for label, value in percentiles.items():
        fig.add_vline(
            x=value,
            line_dash="dash",
            line_color=colors[label],
            annotation_text=f"{label}: {value:.2f}ms",
            annotation_position="top"
        )

    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Latency (ms)",
        yaxis_title="Frequency",
        showlegend=True,
        hovermode='closest',
        template='plotly_white',
        width=1000,
        height=600
    )

    # Add statistics annotation
    stats_text = (
        f"Mean: {stats['mean']:.2f}ms<br>"
        f"Median: {stats['median']:.2f}ms<br>"
        f"Std Dev: {stats['std']:.2f}ms<br>"
        f"Min: {stats['min']:.2f}ms<br>"
        f"Max: {stats['max']:.2f}ms"
    )

    fig.add_annotation(
        text=stats_text,
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        xanchor='right', yanchor='top',
        showarrow=False,
        bordercolor="black",
        borderwidth=1,
        bgcolor="white",
        opacity=0.8
    )

    # Save if output path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_file))
        print(f"✅ Latency histogram saved: {output_file}")

    return fig


def measure_inference_latency(
    model,
    X_test: np.ndarray,
    n_iterations: int = 1000,
    batch_size: int = 1
) -> np.ndarray:
    """
    Measure inference latency for a model

    Args:
        model: Model with predict_proba method
        X_test: Test data
        n_iterations: Number of inference iterations
        batch_size: Batch size for inference

    Returns:
        Array of latencies in milliseconds
    """
    tracker = LatencyTracker()

    for i in range(n_iterations):
        # Sample batch
        if batch_size == 1:
            idx = np.random.randint(0, len(X_test))
            X_batch = X_test[idx:idx+1]
        else:
            idx = np.random.randint(0, len(X_test) - batch_size + 1)
            X_batch = X_test[idx:idx+batch_size]

        # Measure inference time
        tracker.start()
        try:
            _ = model.predict_proba(X_batch)
        except AttributeError:
            _ = model.predict(X_batch)
        tracker.stop()

    return tracker.get_latencies()