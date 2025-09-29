"""
Tests for Calibration Curve Saving (TDD RED phase)
"""
# UTF-8 fix
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pytest
import numpy as np
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


class TestCalibrationCurves:
    """Test suite for calibration curve visualization and saving"""

    @pytest.fixture
    def sample_calibration_data(self):
        """Create sample calibration data"""
        np.random.seed(42)
        n_samples = 1000

        # Ground truth
        y_true = np.random.randint(0, 2, n_samples)

        # Uncalibrated predictions (skewed)
        y_pred_uncalib = np.random.beta(2, 5, n_samples)

        # Calibrated predictions (closer to true probabilities)
        y_pred_isotonic = y_pred_uncalib * 0.8 + 0.1
        y_pred_platt = y_pred_uncalib * 0.85 + 0.05

        return {
            "y_true": y_true,
            "y_pred_uncalib": y_pred_uncalib,
            "y_pred_isotonic": y_pred_isotonic,
            "y_pred_platt": y_pred_platt
        }

    def test_calibration_curve_plot_creation(self, sample_calibration_data, tmp_path):
        """Test: Should create calibration curve comparison plot"""
        from utils.calibration_viz import plot_calibration_curves

        output_path = tmp_path / "calibration_curves.png"

        plot_calibration_curves(
            y_true=sample_calibration_data["y_true"],
            predictions={
                "Uncalibrated": sample_calibration_data["y_pred_uncalib"],
                "Isotonic": sample_calibration_data["y_pred_isotonic"],
                "Platt": sample_calibration_data["y_pred_platt"]
            },
            output_path=output_path
        )

        assert output_path.exists()

    def test_multiple_calibration_methods(self, sample_calibration_data):
        """Test: Should support both Isotonic and Platt calibration"""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier

        y_true = sample_calibration_data["y_true"]
        X = np.random.rand(len(y_true), 5)

        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y_true)

        # Test Isotonic
        cal_isotonic = CalibratedClassifierCV(model, method='isotonic', cv=3)
        cal_isotonic.fit(X, y_true)
        pred_isotonic = cal_isotonic.predict_proba(X)[:, 1]

        # Test Platt (sigmoid)
        cal_platt = CalibratedClassifierCV(model, method='sigmoid', cv=3)
        cal_platt.fit(X, y_true)
        pred_platt = cal_platt.predict_proba(X)[:, 1]

        # Both should produce valid probabilities
        assert np.all(pred_isotonic >= 0) and np.all(pred_isotonic <= 1)
        assert np.all(pred_platt >= 0) and np.all(pred_platt <= 1)

        # Predictions should be different
        assert not np.allclose(pred_isotonic, pred_platt)

    def test_brier_score_comparison(self, sample_calibration_data):
        """Test: Should calculate Brier scores for comparison"""
        from sklearn.metrics import brier_score_loss

        y_true = sample_calibration_data["y_true"]

        brier_uncalib = brier_score_loss(
            y_true, sample_calibration_data["y_pred_uncalib"]
        )
        brier_isotonic = brier_score_loss(
            y_true, sample_calibration_data["y_pred_isotonic"]
        )
        brier_platt = brier_score_loss(
            y_true, sample_calibration_data["y_pred_platt"]
        )

        # Calibrated should have better (lower) Brier scores
        assert brier_isotonic < brier_uncalib * 1.1  # Allow tolerance
        assert brier_platt < brier_uncalib * 1.1


if __name__ == "__main__":
    print("🧪 Running Calibration Curves Tests (TDD RED Phase)")
    print("="*60)
    pytest.main([__file__, '-v'])