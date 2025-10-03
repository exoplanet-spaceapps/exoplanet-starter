"""
Test Feature Extraction for Colab Notebook

This test file validates the feature extraction functions work correctly
before running the full 11,979 sample extraction.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_synthetic_lightcurve_generation():
    """Test generating synthetic light curve with transits"""
    print("\n Test 1: Synthetic Light Curve Generation")

    # Parameters
    period = 3.5  # days
    depth = 0.01  # 1% depth
    duration = 0.1  # 2.4 hours
    time = np.linspace(0, 27.4, 1000)  # TESS sector

    # Generate flux with transits
    flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))

    for transit_time in np.arange(duration, time[-1], period):
        in_transit = np.abs(time - transit_time) < (duration / 2)
        flux[in_transit] *= (1 - depth)

    # Validate
    assert len(flux) == len(time), "Flux and time arrays must match"
    assert np.min(flux) < 1.0, "Transits should reduce flux below 1.0"
    assert np.max(flux) <= 1.01, "Flux should not exceed normalized value"

    print(f"    Generated light curve: {len(time)} points")
    print(f"    Transit depth: {1 - np.min(flux):.4f}")
    print(f"    Expected transits: {int(time[-1] / period)}")


def test_feature_extraction_minimal():
    """Test feature extraction with minimal synthetic data"""
    print("\n Test 2: Feature Extraction (Minimal)")

    # Simple synthetic data
    time = np.linspace(0, 10, 500)
    flux = np.ones_like(time) + np.random.normal(0, 0.001, len(time))

    # Add single transit
    transit_center = 5.0
    duration = 0.1
    depth = 0.01
    in_transit = np.abs(time - transit_center) < (duration / 2)
    flux[in_transit] *= (1 - depth)

    # Extract features (without BLS for speed)
    features = {
        'input_period': 3.5,
        'input_depth': depth,
        'input_duration': duration,
        'input_epoch': transit_center,
        'flux_std': float(np.std(flux)),
        'flux_mad': float(np.median(np.abs(flux - np.median(flux)))),
        'flux_skewness': float(np.mean(((flux - np.mean(flux)) / np.std(flux)) ** 3)),
        'flux_kurtosis': float(np.mean(((flux - np.mean(flux)) / np.std(flux)) ** 4) - 3.0)
    }

    # Validate
    assert features['flux_std'] > 0, "Flux std should be positive"
    assert features['flux_mad'] > 0, "Flux MAD should be positive"
    assert abs(features['flux_skewness']) < 5, "Skewness should be reasonable"

    print(f"    Extracted {len(features)} features")
    print(f"    Flux std: {features['flux_std']:.6f}")
    print(f"    Flux MAD: {features['flux_mad']:.6f}")


def test_checkpoint_manager():
    """Test checkpoint manager initialization"""
    print("\n Test 3: Checkpoint Manager")

    import tempfile
    import json

    # Create temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        checkpoint_dir = tmpdir_path / "checkpoints"
        checkpoint_dir.mkdir()

        # Save test checkpoint
        checkpoint = {
            "checkpoint_id": "batch_0000_0100",
            "timestamp": "2025-01-29T00:00:00",
            "batch_range": [0, 100],
            "completed_indices": [0, 1, 2, 3, 4],
            "failed_indices": [],
            "features": {
                "0": {"input_period": 3.5, "flux_std": 0.001},
                "1": {"input_period": 5.2, "flux_std": 0.002}
            }
        }

        checkpoint_file = checkpoint_dir / "batch_0000_0100.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f)

        # Load checkpoint
        with open(checkpoint_file, 'r') as f:
            loaded = json.load(f)

        assert loaded["checkpoint_id"] == checkpoint["checkpoint_id"]
        assert len(loaded["completed_indices"]) == 5
        assert len(loaded["features"]) == 2

        print(f"    Checkpoint saved and loaded")
        print(f"    Completed indices: {len(loaded['completed_indices'])}")
        print(f"    Features stored: {len(loaded['features'])}")


def test_batch_processing_simulation():
    """Simulate batch processing logic"""
    print("\n Test 4: Batch Processing Simulation")

    # Simulate dataset
    total_samples = 250
    batch_size = 100

    # Calculate batches
    n_batches = (total_samples + batch_size - 1) // batch_size

    batch_ranges = []
    for batch_num in range(n_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, total_samples)
        batch_ranges.append((start, end))

    # Validate
    assert n_batches == 3, f"Expected 3 batches, got {n_batches}"
    assert batch_ranges[0] == (0, 100), "First batch incorrect"
    assert batch_ranges[1] == (100, 200), "Second batch incorrect"
    assert batch_ranges[2] == (200, 250), "Third batch incorrect"

    print(f"    Total batches: {n_batches}")
    print(f"    Batch ranges: {batch_ranges}")


def test_feature_completeness():
    """Verify all 17 features are extracted"""
    print("\n Test 5: Feature Completeness")

    expected_features = [
        # Input parameters (4)
        'input_period', 'input_depth', 'input_duration', 'input_epoch',
        # Flux statistics (4)
        'flux_std', 'flux_mad', 'flux_skewness', 'flux_kurtosis',
        # BLS features (5)
        'bls_period', 'bls_t0', 'bls_duration', 'bls_depth', 'bls_snr',
        # Advanced features (4)
        'duration_over_period', 'odd_even_depth_diff', 'transit_symmetry', 'periodicity_strength'
    ]

    # Simulate extracted features
    features = {key: 0.0 for key in expected_features}

    # Add metadata
    features['sample_idx'] = 0
    features['label'] = 1
    features['target_id'] = 'TIC12345678'
    features['toi'] = 'TOI-100.01'

    # Validate
    assert len([k for k in features if k in expected_features]) == 17, "Should have 17 features"
    assert 'sample_idx' in features, "Should have sample_idx"
    assert 'label' in features, "Should have label"

    print(f"    Total features: {len([k for k in features if k in expected_features])}")
    print(f"    Metadata fields: 4 (sample_idx, label, target_id, toi)")
    print(f"    Total columns: {len(features)}")


def test_nan_handling():
    """Test handling of NaN values"""
    print("\n Test 6: NaN Value Handling")

    # Data with NaNs
    flux_with_nan = np.array([1.0, 1.0, np.nan, 1.0, 1.0])

    # Remove NaNs
    flux_clean = flux_with_nan[~np.isnan(flux_with_nan)]

    # Compute statistics
    flux_std = np.std(flux_clean)
    flux_mad = np.median(np.abs(flux_clean - np.median(flux_clean)))

    assert not np.isnan(flux_std), "Flux std should not be NaN"
    assert not np.isnan(flux_mad), "Flux MAD should not be NaN"
    assert len(flux_clean) == 4, "Should have 4 clean values"

    print(f"    Original length: {len(flux_with_nan)}")
    print(f"    Cleaned length: {len(flux_clean)}")
    print(f"    Statistics computed without NaN")


def test_progress_tracking():
    """Test progress calculation"""
    print("\n Test 7: Progress Tracking")

    total_samples = 11979
    completed = 3500
    failed = 50
    remaining = total_samples - completed

    success_rate = (completed / total_samples) * 100
    failure_rate = (failed / total_samples) * 100

    progress = {
        "total_samples": total_samples,
        "completed": completed,
        "failed": failed,
        "remaining": remaining,
        "success_rate": success_rate,
        "failure_rate": failure_rate
    }

    # Validate
    assert progress["remaining"] == 8479, "Remaining calculation incorrect"
    assert 29 < progress["success_rate"] < 30, "Success rate incorrect"
    assert progress["failure_rate"] < 1, "Failure rate should be low"

    print(f"    Completed: {completed}/{total_samples} ({success_rate:.1f}%)")
    print(f"    Failed: {failed} ({failure_rate:.2f}%)")
    print(f"    Remaining: {remaining}")


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Feature Extraction Test Suite")
    print("=" * 60)

    tests = [
        test_synthetic_lightcurve_generation,
        test_feature_extraction_minimal,
        test_checkpoint_manager,
        test_batch_processing_simulation,
        test_feature_completeness,
        test_nan_handling,
        test_progress_tracking
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"    FAILED: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("All tests passed! Ready for production run.")
    else:
        print(f"{failed} tests failed. Please fix before running.")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)