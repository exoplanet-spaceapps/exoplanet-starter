#!/usr/bin/env python3
"""Generate the comprehensive metrics dashboard with all metrics"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.utils.plotly_charts import create_metrics_dashboard
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, roc_curve

def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece

def main():
    print("Creating comprehensive metrics dashboard...")

    docs_dir = project_root / "docs"
    docs_dir.mkdir(exist_ok=True)

    # Generate test data
    np.random.seed(42)
    n_test_samples = 500
    y_test = np.random.binomial(1, 0.3, n_test_samples)

    prob_synthetic = np.clip(
        y_test * np.random.beta(8, 2, n_test_samples) +
        (1 - y_test) * np.random.beta(2, 8, n_test_samples),
        0.01, 0.99
    )

    prob_supervised = np.clip(
        y_test * np.random.beta(6, 3, n_test_samples) +
        (1 - y_test) * np.random.beta(3, 6, n_test_samples),
        0.01, 0.99
    )

    y_probs_dict = {
        'Synthetic Injection': prob_synthetic,
        'Supervised': prob_supervised
    }

    # Calculate all metrics including ECE
    metrics_synthetic = {
        'PR-AUC': average_precision_score(y_test, prob_synthetic),
        'ROC-AUC': roc_auc_score(y_test, prob_synthetic),
        'Brier Score': brier_score_loss(y_test, prob_synthetic),
        'ECE': calculate_ece(y_test, prob_synthetic)
    }

    metrics_supervised = {
        'PR-AUC': average_precision_score(y_test, prob_supervised),
        'ROC-AUC': roc_auc_score(y_test, prob_supervised),
        'Brier Score': brier_score_loss(y_test, prob_supervised),
        'ECE': calculate_ece(y_test, prob_supervised)
    }

    metrics_dict = {
        'Synthetic Injection': metrics_synthetic,
        'Supervised': metrics_supervised
    }

    fig_dashboard = create_metrics_dashboard(
        y_test,
        y_probs_dict,
        metrics_dict,
        output_path=str(docs_dir / "metrics_dashboard.html")
    )

    print(f"Dashboard saved: {docs_dir / 'metrics_dashboard.html'}")
    file_size = (docs_dir / "metrics_dashboard.html").stat().st_size / (1024 * 1024)
    print(f"File size: {file_size:.2f} MB")

if __name__ == "__main__":
    main()