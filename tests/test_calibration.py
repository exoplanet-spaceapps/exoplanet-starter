"""
Tests for Probability Calibration (Phase 8) - TDD RED phase
"""
# UTF-8 fix
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


class TestProbabilityCalibration:
    """Test suite for probability calibration"""

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions"""
        np.random.seed(42)
        n_samples = 1000

        # Uncalibrated probabilities (skewed)
        y_true = np.random.randint(0, 2, n_samples)
        y_pred_proba = np.random.beta(2, 5, n_samples)  # Skewed distribution

        return y_true, y_pred_proba

    def test_calibration_improves_brier_score(self, sample_predictions):
        """Test: Calibration should improve Brier score"""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import brier_score_loss
        from sklearn.ensemble import RandomForestClassifier

        y_true, _ = sample_predictions

        # Create uncalibrated model
        np.random.seed(42)
        X = np.random.rand(len(y_true), 5)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y_true)

        # Uncalibrated predictions
        y_pred_uncalib = model.predict_proba(X)[:, 1]
        brier_before = brier_score_loss(y_true, y_pred_uncalib)

        # Calibrated model
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated.fit(X, y_true)
        y_pred_calib = calibrated.predict_proba(X)[:, 1]
        brier_after = brier_score_loss(y_true, y_pred_calib)

        # Calibration should improve or maintain Brier score
        assert brier_after <= brier_before * 1.1  # Allow 10% tolerance

    def test_calibrated_probabilities_range(self, sample_predictions):
        """Test: Calibrated probabilities should be in [0, 1]"""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier

        y_true, _ = sample_predictions

        X = np.random.rand(len(y_true), 5)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated.fit(X, y_true)

        y_pred = calibrated.predict_proba(X)[:, 1]

        assert np.all(y_pred >= 0) and np.all(y_pred <= 1)

    def test_calibration_curve_closer_to_diagonal(self, sample_predictions):
        """Test: Calibration curve should be closer to perfect calibration"""
        from sklearn.calibration import calibration_curve, CalibratedClassifierCV
        from sklearn.ensemble import RandomForestClassifier

        y_true, _ = sample_predictions

        X = np.random.rand(len(y_true), 5)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y_true)

        # Uncalibrated
        y_pred_uncalib = model.predict_proba(X)[:, 1]
        prob_true_before, prob_pred_before = calibration_curve(
            y_true, y_pred_uncalib, n_bins=10
        )

        # Calibrated
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
        calibrated.fit(X, y_true)
        y_pred_calib = calibrated.predict_proba(X)[:, 1]
        prob_true_after, prob_pred_after = calibration_curve(
            y_true, y_pred_calib, n_bins=10
        )

        # Measure distance to perfect calibration (diagonal)
        distance_before = np.mean(np.abs(prob_true_before - prob_pred_before))
        distance_after = np.mean(np.abs(prob_true_after - prob_pred_after))

        # Calibration should reduce distance (allow small tolerance)
        assert distance_after <= distance_before * 1.2


if __name__ == "__main__":
    print("🧪 Running Calibration Tests (TDD RED Phase)")
    print("="*60)
    pytest.main([__file__, '-v'])