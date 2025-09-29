"""
Calibration Curve Visualization (GREEN phase implementation)
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def plot_calibration_curves(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_path: Path,
    n_bins: int = 10
) -> None:
    """
    Create calibration curve comparison plot for multiple methods

    Args:
        y_true: True binary labels
        predictions: Dict mapping method names to predicted probabilities
                    e.g., {"Uncalibrated": pred1, "Isotonic": pred2, "Platt": pred3}
        output_path: Path to save plot (e.g., reports/calibration_curves.png)
        n_bins: Number of bins for calibration curve
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ['red', 'blue', 'green', 'purple', 'orange']

    # Left plot: Calibration curves
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)

    for i, (method, y_pred) in enumerate(predictions.items()):
        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
        brier = brier_score_loss(y_true, y_pred)

        ax1.plot(
            prob_pred, prob_true,
            marker='o',
            linewidth=2,
            color=colors[i % len(colors)],
            label=f'{method} (Brier: {brier:.4f})'
        )

    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Calibration Curves Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Right plot: Reliability diagram with histograms
    for i, (method, y_pred) in enumerate(predictions.items()):
        ax2.hist(
            y_pred,
            bins=20,
            alpha=0.5,
            label=method,
            color=colors[i % len(colors)]
        )

    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Calibration curves saved to: {output_path}")


def compare_calibration_methods(
    y_true: np.ndarray,
    y_pred_uncalib: np.ndarray,
    y_pred_isotonic: np.ndarray,
    y_pred_platt: np.ndarray
) -> Dict[str, float]:
    """
    Compare Brier scores for different calibration methods

    Args:
        y_true: True binary labels
        y_pred_uncalib: Uncalibrated predictions
        y_pred_isotonic: Isotonic calibrated predictions
        y_pred_platt: Platt calibrated predictions

    Returns:
        Dict of Brier scores
    """
    scores = {
        "uncalibrated": brier_score_loss(y_true, y_pred_uncalib),
        "isotonic": brier_score_loss(y_true, y_pred_isotonic),
        "platt": brier_score_loss(y_true, y_pred_platt)
    }

    # Calculate improvements
    scores["isotonic_improvement"] = (
        (scores["uncalibrated"] - scores["isotonic"]) / scores["uncalibrated"] * 100
    )
    scores["platt_improvement"] = (
        (scores["uncalibrated"] - scores["platt"]) / scores["uncalibrated"] * 100
    )

    return scores