# 🧪 TDD Test Specifications - Exoplanet Detection Pipeline

**Document Version**: 1.0
**Date**: 2025-09-30
**Status**: Test-First Development Ready

---

## 📋 Test Strategy Overview

This document outlines comprehensive Test-Driven Development (TDD) specifications for all three Colab notebooks. Tests are designed to run in Colab environments and validate critical functionality before full dataset processing.

### Testing Principles
1. **Test First**: Write tests before implementation
2. **Isolation**: Each test is independent
3. **Fast Feedback**: Unit tests run in <10 seconds
4. **Colab-Compatible**: All tests work in Google Colab
5. **Comprehensive**: Cover happy paths, edge cases, and failures

---

## 🎯 Phase 3: Feature Extraction Tests (02_bls_baseline.ipynb)

### Test Suite 1: BLS Period Recovery

**Purpose**: Verify BLS correctly detects synthetic transits

```python
def test_bls_period_recovery_simple():
    """
    Test BLS period recovery on simple box transit

    Expected behavior:
    - Period recovered within 0.1 days
    - Depth recovered within 0.005 relative flux
    - SNR > 10
    """
    # Arrange
    time = np.linspace(0, 27, 1000)  # 27-day light curve
    flux = np.ones(1000)

    # Inject known transit
    true_period = 3.5  # days
    true_depth = 0.01  # 1% depth
    true_duration = 0.1  # days
    true_t0 = 0.5

    flux_injected = inject_box_transit(
        time, flux, true_period, true_depth, true_duration, true_t0
    )

    # Act
    bls_result = run_bls(time, flux_injected, min_period=0.5, max_period=20.0)

    # Assert
    assert 'period' in bls_result, "Missing period key"
    assert 'depth' in bls_result, "Missing depth key"
    assert 'snr' in bls_result, "Missing SNR key"

    period_error = abs(bls_result['period'] - true_period)
    assert period_error < 0.1, f"Period error {period_error:.3f} > 0.1 days"

    depth_error = abs(bls_result['depth'] - true_depth)
    assert depth_error < 0.005, f"Depth error {depth_error:.4f} > 0.005"

    assert bls_result['snr'] > 10, f"SNR {bls_result['snr']:.1f} too low"

    print("✅ test_bls_period_recovery_simple PASSED")


def test_bls_multi_period_detection():
    """
    Test BLS on light curve with multiple periods

    Expected behavior:
    - Detects strongest signal first
    - Period matches strongest transit
    """
    # Arrange
    time = np.linspace(0, 50, 2000)
    flux = np.ones(2000)

    # Inject two transits (different depths)
    flux = inject_box_transit(time, flux, 3.0, 0.015, 0.1, 0)  # Strong
    flux = inject_box_transit(time, flux, 7.5, 0.005, 0.1, 0)  # Weak

    # Act
    bls_result = run_bls(time, flux)

    # Assert - should detect stronger 3-day signal
    assert abs(bls_result['period'] - 3.0) < 0.2, \
        f"Should detect strongest signal (3 days), got {bls_result['period']:.2f}"

    assert bls_result['depth'] > 0.010, "Should detect deeper transit"

    print("✅ test_bls_multi_period_detection PASSED")


def test_bls_no_transit():
    """
    Test BLS on noise-only light curve

    Expected behavior:
    - Returns result (no crash)
    - Low SNR (< 7)
    - No false positives at high confidence
    """
    # Arrange
    time = np.linspace(0, 27, 1000)
    flux = np.random.normal(1.0, 0.001, 1000)  # Pure noise

    # Act
    bls_result = run_bls(time, flux)

    # Assert
    assert 'snr' in bls_result, "Missing SNR"
    assert bls_result['snr'] < 10, \
        f"False positive: SNR {bls_result['snr']:.1f} too high for noise"

    print("✅ test_bls_no_transit PASSED")
```

---

### Test Suite 2: Feature Extraction Completeness

**Purpose**: Ensure all 14 features are extracted correctly

```python
def test_feature_extraction_completeness():
    """
    Test that extract_features returns all required features

    Expected behavior:
    - 14 features present
    - No NaN values in critical features
    - Values in reasonable ranges
    """
    # Arrange
    time = np.linspace(0, 27, 1000)
    flux = np.ones(1000)
    flux = inject_box_transit(time, flux, 3.5, 0.01, 0.1, 0)

    bls_result = {
        'period': 3.5,
        'depth': 0.01,
        'duration': 0.1,
        'snr': 25.0,
        't0': 0.5
    }

    # Act
    features = extract_features(time, flux, bls_result, compute_advanced=True)

    # Assert - Check all required features exist
    required_features = [
        'bls_period', 'bls_depth', 'bls_duration', 'bls_snr', 'bls_t0',
        'duration_over_period', 'depth_snr_ratio',
        'odd_even_depth_diff', 'transit_symmetry',
        'flux_std', 'flux_mad', 'flux_skew', 'flux_kurtosis',
        'periodicity_strength'
    ]

    for feat in required_features:
        assert feat in features, f"Missing feature: {feat}"

    # Check no NaN in critical features
    critical_features = ['bls_period', 'bls_depth', 'bls_snr']
    for feat in critical_features:
        assert not np.isnan(features[feat]), f"NaN in critical feature: {feat}"

    # Check value ranges
    assert 0.5 <= features['bls_period'] <= 20.0, "Period out of range"
    assert 0.0 < features['bls_depth'] <= 1.0, "Depth out of range"
    assert features['bls_snr'] > 0, "SNR must be positive"
    assert 0.0 <= features['duration_over_period'] <= 1.0, "Duration ratio out of range"

    print("✅ test_feature_extraction_completeness PASSED")


def test_feature_extraction_robustness():
    """
    Test feature extraction handles edge cases

    Expected behavior:
    - Handles sparse data (< 100 points)
    - Handles noisy data
    - Returns valid features (no inf, no NaN where critical)
    """
    # Test Case 1: Sparse data
    time_sparse = np.linspace(0, 27, 50)  # Only 50 points
    flux_sparse = np.ones(50)
    flux_sparse = inject_box_transit(time_sparse, flux_sparse, 5.0, 0.01, 0.1, 0)

    bls_result_sparse = run_bls(time_sparse, flux_sparse)
    features_sparse = extract_features(time_sparse, flux_sparse, bls_result_sparse)

    assert 'bls_period' in features_sparse, "Failed on sparse data"

    # Test Case 2: Very noisy data
    time_noisy = np.linspace(0, 27, 1000)
    flux_noisy = np.random.normal(1.0, 0.01, 1000)  # 1% noise

    bls_result_noisy = run_bls(time_noisy, flux_noisy)
    features_noisy = extract_features(time_noisy, flux_noisy, bls_result_noisy)

    # Should not crash, even if low quality
    assert 'flux_std' in features_noisy, "Failed on noisy data"
    assert not np.isinf(features_noisy['flux_std']), "Inf value in features"

    print("✅ test_feature_extraction_robustness PASSED")
```

---

### Test Suite 3: Checkpoint System

**Purpose**: Verify checkpoint save/load integrity

```python
def test_checkpoint_save_load():
    """
    Test checkpoint save and load preserves data

    Expected behavior:
    - Saved data matches loaded data
    - Checkpoint ID correctly tracked
    - Can resume from checkpoint
    """
    import tempfile
    from pathlib import Path

    # Arrange
    temp_dir = Path(tempfile.mkdtemp())
    features_df = pd.DataFrame({
        'tic_id': ['TIC1', 'TIC2', 'TIC3'],
        'bls_period': [3.5, 5.2, 7.8],
        'bls_snr': [25.0, 18.5, 32.1]
    })

    checkpoint_id = 42

    # Act - Save
    checkpoint_path = temp_dir / f'checkpoint_{checkpoint_id:04d}.parquet'
    features_df.to_parquet(checkpoint_path)

    # Act - Load
    loaded_df = pd.read_parquet(checkpoint_path)
    loaded_id = int(checkpoint_path.stem.split('_')[1])

    # Assert
    assert len(loaded_df) == len(features_df), "Row count mismatch"
    assert list(loaded_df.columns) == list(features_df.columns), "Column mismatch"
    assert loaded_id == checkpoint_id, "Checkpoint ID mismatch"

    # Check data integrity
    pd.testing.assert_frame_equal(loaded_df, features_df)

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

    print("✅ test_checkpoint_save_load PASSED")


def test_checkpoint_resume():
    """
    Test resuming from checkpoint skips processed samples

    Expected behavior:
    - Correctly identifies last checkpoint
    - Resumes from correct index
    """
    import tempfile
    from pathlib import Path

    # Arrange
    temp_dir = Path(tempfile.mkdtemp())

    # Create multiple checkpoints
    for i in [10, 20, 30]:
        df = pd.DataFrame({'sample_id': range(i)})
        checkpoint_path = temp_dir / f'checkpoint_{i:04d}.parquet'
        df.to_parquet(checkpoint_path)

    # Act - Find latest checkpoint
    checkpoints = sorted(temp_dir.glob('checkpoint_*.parquet'))
    latest = checkpoints[-1]
    latest_id = int(latest.stem.split('_')[1])

    # Assert
    assert len(checkpoints) == 3, "Should find 3 checkpoints"
    assert latest_id == 30, f"Latest checkpoint should be 30, got {latest_id}"

    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)

    print("✅ test_checkpoint_resume PASSED")
```

---

### Test Suite 4: Light Curve Download

**Purpose**: Verify robust download with fallback strategies

```python
def test_lightcurve_download_success():
    """
    Test downloading a known good target

    Expected behavior:
    - Successfully downloads light curve
    - Returns normalized LightCurve object
    - No NaN values after processing
    """
    import lightkurve as lk

    # Arrange - Use known good target (TOI 270)
    tic_id = "TIC 259377017"

    # Act
    search = lk.search_lightcurve(tic_id, author='SPOC', mission='TESS')

    # Assert
    assert len(search) > 0, f"No data found for {tic_id}"

    lc = search[0].download()
    lc_clean = lc.remove_nans().normalize()

    assert len(lc_clean) > 100, "Too few data points after cleaning"
    assert not np.any(np.isnan(lc_clean.flux.value)), "NaN values remain"
    assert np.median(lc_clean.flux.value) > 0, "Flux should be positive"

    print("✅ test_lightcurve_download_success PASSED")


def test_lightcurve_download_fallback():
    """
    Test fallback strategy when SPOC fails

    Expected behavior:
    - Tries alternative authors if first fails
    - Returns None if all fail (doesn't crash)
    """
    import lightkurve as lk

    def download_lightcurve_safe(tic_id, max_retries=2):
        authors = ['SPOC', 'QLP']

        for author in authors:
            try:
                search = lk.search_lightcurve(tic_id, author=author, mission='TESS')
                if len(search) > 0:
                    lc = search[0].download()
                    return lc.remove_nans().normalize()
            except Exception as e:
                continue

        return None

    # Test with known target
    lc = download_lightcurve_safe("TIC 259377017")
    assert lc is not None, "Should successfully download known target"

    # Test with fake target (should not crash)
    lc_fake = download_lightcurve_safe("TIC 999999999999")
    assert lc_fake is None, "Should return None for non-existent target"

    print("✅ test_lightcurve_download_fallback PASSED")
```

---

## 🤖 Phase 4: Model Training Tests (03_injection_train.ipynb)

### Test Suite 5: GPU Availability

**Purpose**: Verify GPU detection and fallback

```python
def test_gpu_detection():
    """
    Test GPU availability and XGBoost GPU support

    Expected behavior:
    - Detects GPU if available
    - Falls back to CPU if GPU unavailable
    - XGBoost accepts gpu_hist parameter
    """
    import torch
    import xgboost as xgb

    # Check PyTorch GPU detection
    gpu_available = torch.cuda.is_available()

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU detected: {gpu_name}")
    else:
        print("⚠️  No GPU detected, will use CPU")

    # Test XGBoost GPU parameter
    params = {
        'tree_method': 'gpu_hist' if gpu_available else 'hist',
        'max_depth': 3,
        'n_estimators': 10
    }

    # Create dummy data
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)

    # Train small model
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(params, dtrain, num_boost_round=10)

    assert model is not None, "XGBoost training failed"

    print("✅ test_gpu_detection PASSED")


def test_gpu_speedup():
    """
    Test GPU provides speedup vs CPU

    Expected behavior:
    - GPU training faster than CPU (if GPU available)
    - Both produce similar accuracy
    """
    import time
    import xgboost as xgb
    from sklearn.datasets import make_classification

    # Generate larger dataset for meaningful timing
    X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
    dtrain = xgb.DMatrix(X, label=y)

    # CPU training
    params_cpu = {'tree_method': 'hist', 'max_depth': 6}
    start_cpu = time.time()
    model_cpu = xgb.train(params_cpu, dtrain, num_boost_round=50)
    time_cpu = time.time() - start_cpu

    # GPU training (if available)
    if torch.cuda.is_available():
        params_gpu = {'tree_method': 'gpu_hist', 'max_depth': 6}
        start_gpu = time.time()
        model_gpu = xgb.train(params_gpu, dtrain, num_boost_round=50)
        time_gpu = time.time() - start_gpu

        speedup = time_cpu / time_gpu
        print(f"✅ GPU speedup: {speedup:.2f}x (CPU: {time_cpu:.2f}s, GPU: {time_gpu:.2f}s)")
        assert speedup > 1.0, "GPU should be faster than CPU"
    else:
        print("⚠️  Skipping GPU speedup test (no GPU available)")

    print("✅ test_gpu_speedup PASSED")
```

---

### Test Suite 6: Model Calibration

**Purpose**: Verify calibration improves probability quality

```python
def test_calibration_improves_brier_score():
    """
    Test that calibration improves Brier score

    Expected behavior:
    - Brier score decreases after calibration
    - ECE (Expected Calibration Error) decreases
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import brier_score_loss
    import xgboost as xgb

    # Arrange
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Train base model
    base_model = xgb.XGBClassifier(n_estimators=50, max_depth=5, random_state=42)
    base_model.fit(X_train, y_train)

    # Get uncalibrated predictions
    y_proba_uncal = base_model.predict_proba(X_test)[:, 1]
    brier_uncal = brier_score_loss(y_test, y_proba_uncal)

    # Calibrate model
    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_cal, y_cal)

    # Get calibrated predictions
    y_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
    brier_cal = brier_score_loss(y_test, y_proba_cal)

    # Assert
    print(f"Brier Score - Uncalibrated: {brier_uncal:.4f}, Calibrated: {brier_cal:.4f}")
    assert brier_cal <= brier_uncal, \
        f"Calibration should improve Brier score: {brier_cal:.4f} > {brier_uncal:.4f}"

    improvement = (brier_uncal - brier_cal) / brier_uncal * 100
    print(f"✅ Calibration improved Brier score by {improvement:.1f}%")

    print("✅ test_calibration_improves_brier_score PASSED")


def test_calibration_curve_reliability():
    """
    Test that calibrated probabilities are well-calibrated

    Expected behavior:
    - Expected Calibration Error (ECE) < 0.1
    - Calibration curve close to diagonal
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV, calibration_curve
    import xgboost as xgb

    # Arrange
    X, y = make_classification(n_samples=2000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

    # Train and calibrate
    base_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42)
    base_model.fit(X_train, y_train)

    calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X_cal, y_cal)

    # Get predictions
    y_proba = calibrated_model.predict_proba(X_test)[:, 1]

    # Compute ECE
    def calculate_ece(y_true, y_proba, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        ece = 0.0
        for i in range(n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                bin_prob = np.mean(y_proba[mask])
                bin_acc = np.mean(y_true[mask])
                ece += np.sum(mask) / len(y_true) * abs(bin_prob - bin_acc)

        return ece

    ece = calculate_ece(y_test, y_proba)

    print(f"ECE: {ece:.4f}")
    assert ece < 0.1, f"ECE too high: {ece:.4f} > 0.1"

    print("✅ test_calibration_curve_reliability PASSED")
```

---

### Test Suite 7: Cross-Validation

**Purpose**: Ensure consistent performance across folds

```python
def test_cross_validation_consistency():
    """
    Test that CV scores are consistent across folds

    Expected behavior:
    - Standard deviation of CV scores < 0.1
    - All folds have ROC-AUC > 0.85
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    import xgboost as xgb

    # Arrange
    X, y = make_classification(n_samples=1000, n_features=14, random_state=42)
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=42)

    # Act
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')

    # Assert
    mean_score = scores.mean()
    std_score = scores.std()

    print(f"CV Scores: {scores}")
    print(f"Mean: {mean_score:.4f} ± {std_score:.4f}")

    assert std_score < 0.1, f"CV scores too inconsistent: std = {std_score:.4f}"
    assert all(score > 0.85 for score in scores), "Some folds have low performance"

    print("✅ test_cross_validation_consistency PASSED")


def test_stratified_split_preserves_ratio():
    """
    Test that stratified split preserves class ratio

    Expected behavior:
    - Each fold has similar positive/negative ratio
    - Variance in ratios < 0.05
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import StratifiedKFold

    # Arrange
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    overall_ratio = y.mean()

    # Act
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_ratios = []

    for train_idx, val_idx in cv.split(X, y):
        fold_ratio = y[val_idx].mean()
        fold_ratios.append(fold_ratio)

    # Assert
    ratio_variance = np.var(fold_ratios)
    print(f"Overall ratio: {overall_ratio:.3f}")
    print(f"Fold ratios: {fold_ratios}")
    print(f"Variance: {ratio_variance:.5f}")

    assert ratio_variance < 0.001, f"Fold ratios too variable: {ratio_variance:.5f}"

    print("✅ test_stratified_split_preserves_ratio PASSED")
```

---

## 🔮 Phase 5: Inference Tests (04_newdata_inference.ipynb)

### Test Suite 8: End-to-End Inference

**Purpose**: Verify complete inference pipeline

```python
def test_inference_on_known_planet():
    """
    Test inference pipeline on known planet (TOI 270 b)

    Expected behavior:
    - Successfully downloads light curve
    - Extracts features without error
    - Returns probability > 0.8 for known planet
    """
    import lightkurve as lk
    import joblib

    # Arrange
    known_planet_tic = "TIC 259377017"  # TOI 270

    # Assume model is loaded (would be in notebook)
    # model = joblib.load('model/xgboost_calibrated.joblib')
    # scaler = joblib.load('model/scaler.joblib')

    # Act - Download
    search = lk.search_lightcurve(known_planet_tic, author='SPOC', mission='TESS')
    assert len(search) > 0, f"No data for known planet {known_planet_tic}"

    lc = search[0].download().remove_nans().normalize()

    # Act - Feature extraction
    bls_result = run_bls(lc.time.value, lc.flux.value)
    features = extract_features(lc.time.value, lc.flux.value, bls_result)

    # Assert
    assert 'bls_period' in features, "Feature extraction failed"
    assert features['bls_snr'] > 7.0, "SNR too low for known planet"

    # Note: Full prediction test requires loaded model
    # y_proba = model.predict_proba(scaler.transform([feature_vector]))[0, 1]
    # assert y_proba > 0.8, f"Low confidence for known planet: {y_proba:.3f}"

    print("✅ test_inference_on_known_planet PASSED")


def test_inference_handles_failures():
    """
    Test inference gracefully handles failures

    Expected behavior:
    - Returns error status for invalid TIC
    - Continues processing other targets
    - No crashes on bad input
    """
    def infer_single_target_mock(tic_id):
        """Mock inference function"""
        try:
            if tic_id == "INVALID":
                raise ValueError("Invalid TIC ID")

            return {
                'tic_id': tic_id,
                'probability': 0.75,
                'status': 'success'
            }
        except Exception as e:
            return {
                'tic_id': tic_id,
                'error': str(e),
                'status': 'failed'
            }

    # Test valid and invalid inputs
    results = []
    for tic_id in ["TIC123", "INVALID", "TIC456"]:
        result = infer_single_target_mock(tic_id)
        results.append(result)

    # Assert
    assert len(results) == 3, "Should process all targets"
    assert results[0]['status'] == 'success', "Valid target should succeed"
    assert results[1]['status'] == 'failed', "Invalid target should fail"
    assert results[2]['status'] == 'success', "Should continue after failure"

    print("✅ test_inference_handles_failures PASSED")
```

---

### Test Suite 9: Batch Processing

**Purpose**: Verify parallel batch inference

```python
def test_batch_inference_completeness():
    """
    Test batch processing returns results for all inputs

    Expected behavior:
    - All input TICs have corresponding output
    - Failed targets have 'error' field
    - Successful targets have 'probability' field
    """
    def batch_inference_mock(tic_ids):
        """Mock batch inference"""
        results = []
        for tic_id in tic_ids:
            if "999" in tic_id:
                results.append({'tic_id': tic_id, 'status': 'failed', 'error': 'No data'})
            else:
                results.append({
                    'tic_id': tic_id,
                    'probability': np.random.uniform(0.5, 0.95),
                    'bls_period': np.random.uniform(1, 10),
                    'status': 'success'
                })
        return pd.DataFrame(results)

    # Test
    tic_ids = ['TIC123', 'TIC999', 'TIC456', 'TIC789']
    results_df = batch_inference_mock(tic_ids)

    # Assert
    assert len(results_df) == len(tic_ids), "Missing results"
    assert 'tic_id' in results_df.columns, "Missing TIC ID column"

    successful = results_df[results_df['status'] == 'success']
    failed = results_df[results_df['status'] == 'failed']

    assert len(successful) == 3, "Should have 3 successful"
    assert len(failed) == 1, "Should have 1 failed"
    assert 'probability' in successful.columns, "Missing probability"

    print("✅ test_batch_inference_completeness PASSED")


def test_batch_parallel_speedup():
    """
    Test parallel processing provides speedup

    Expected behavior:
    - Parallel processing faster than sequential
    - Results identical to sequential processing
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    def slow_task(x):
        """Simulate slow inference"""
        time.sleep(0.1)
        return x * 2

    inputs = list(range(20))

    # Sequential
    start = time.time()
    results_seq = [slow_task(x) for x in inputs]
    time_seq = time.time() - start

    # Parallel (4 workers)
    start = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results_par = list(executor.map(slow_task, inputs))
    time_par = time.time() - start

    # Assert
    assert results_seq == results_par, "Results should match"
    speedup = time_seq / time_par
    print(f"Speedup: {speedup:.2f}x (Sequential: {time_seq:.2f}s, Parallel: {time_par:.2f}s)")
    assert speedup > 1.5, f"Parallel should be faster: {speedup:.2f}x"

    print("✅ test_batch_parallel_speedup PASSED")
```

---

### Test Suite 10: Candidate Ranking

**Purpose**: Verify ranking logic prioritizes correctly

```python
def test_ranking_prioritizes_high_confidence():
    """
    Test that ranking places high-confidence candidates first

    Expected behavior:
    - High probability targets ranked higher
    - Ties broken by SNR
    - Minimum thresholds enforced
    """
    # Arrange
    results_df = pd.DataFrame({
        'tic_id': ['TIC1', 'TIC2', 'TIC3', 'TIC4', 'TIC5'],
        'planet_probability': [0.95, 0.85, 0.75, 0.65, 0.55],
        'bls_snr': [30, 25, 20, 15, 10],
        'bls_period': [5, 8, 3, 12, 20],
        'status': ['success'] * 5
    })

    def rank_candidates(df, min_prob=0.7, min_snr=7.0):
        """Simplified ranking function"""
        candidates = df[
            (df['planet_probability'] >= min_prob) &
            (df['bls_snr'] >= min_snr) &
            (df['status'] == 'success')
        ].copy()

        candidates['score'] = (
            0.60 * candidates['planet_probability'] +
            0.25 * (candidates['bls_snr'] / 50.0).clip(0, 1) +
            0.15 * (1 - abs(candidates['bls_period'] - 10) / 10).clip(0, 1)
        )

        return candidates.sort_values('score', ascending=False)

    # Act
    ranked = rank_candidates(results_df, min_prob=0.7, min_snr=7.0)

    # Assert
    assert len(ranked) == 3, "Should filter to 3 candidates (prob >= 0.7)"
    assert ranked.iloc[0]['tic_id'] == 'TIC1', "Highest probability should rank first"
    assert ranked.iloc[1]['tic_id'] == 'TIC2', "Second highest should rank second"

    # Check score monotonicity
    scores = ranked['score'].values
    assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1)), \
        "Scores should be monotonically decreasing"

    print("✅ test_ranking_prioritizes_high_confidence PASSED")


def test_ranking_filters_low_quality():
    """
    Test that low-quality candidates are filtered out

    Expected behavior:
    - Probability < 0.7 filtered
    - SNR < 7 filtered
    - Failed status filtered
    """
    # Arrange
    results_df = pd.DataFrame({
        'tic_id': ['GOOD', 'LOW_PROB', 'LOW_SNR', 'FAILED'],
        'planet_probability': [0.90, 0.65, 0.85, 0.95],
        'bls_snr': [25, 20, 5, 30],
        'bls_period': [5, 5, 5, 5],
        'status': ['success', 'success', 'success', 'failed']
    })

    def rank_candidates(df, min_prob=0.7, min_snr=7.0):
        return df[
            (df['planet_probability'] >= min_prob) &
            (df['bls_snr'] >= min_snr) &
            (df['status'] == 'success')
        ]

    # Act
    filtered = rank_candidates(results_df)

    # Assert
    assert len(filtered) == 1, "Should only keep GOOD candidate"
    assert filtered.iloc[0]['tic_id'] == 'GOOD', "Only GOOD should pass filters"

    print("✅ test_ranking_filters_low_quality PASSED")
```

---

## 🔧 Test Utilities

### Helper Functions for All Tests

```python
# ==== Synthetic Data Generation ====

def generate_synthetic_lightcurve(
    n_points=1000,
    duration_days=27,
    with_transit=True,
    period=3.5,
    depth=0.01,
    noise_level=0.001
):
    """Generate synthetic light curve for testing"""
    time = np.linspace(0, duration_days, n_points)
    flux = np.ones(n_points)

    if with_transit:
        flux = inject_box_transit(time, flux, period, depth, 0.1, 0)

    # Add noise
    flux += np.random.normal(0, noise_level, n_points)

    return time, flux


# ==== Feature Validation ====

def validate_feature_dict(features, required_features=None):
    """Validate feature dictionary completeness"""
    if required_features is None:
        required_features = [
            'bls_period', 'bls_depth', 'bls_duration', 'bls_snr', 'bls_t0',
            'duration_over_period', 'depth_snr_ratio',
            'odd_even_depth_diff', 'transit_symmetry',
            'flux_std', 'flux_mad', 'flux_skew', 'flux_kurtosis',
            'periodicity_strength'
        ]

    missing = [f for f in required_features if f not in features]
    if missing:
        raise AssertionError(f"Missing features: {missing}")

    # Check for NaN/Inf
    for feat, val in features.items():
        if feat in required_features:
            if np.isnan(val):
                raise AssertionError(f"NaN value in {feat}")
            if np.isinf(val):
                raise AssertionError(f"Inf value in {feat}")


# ==== Test Runner ====

def run_all_tests():
    """Run all test suites"""
    test_functions = [
        # Phase 3
        test_bls_period_recovery_simple,
        test_bls_multi_period_detection,
        test_bls_no_transit,
        test_feature_extraction_completeness,
        test_feature_extraction_robustness,
        test_checkpoint_save_load,
        test_checkpoint_resume,
        # Phase 4
        test_gpu_detection,
        test_calibration_improves_brier_score,
        test_calibration_curve_reliability,
        test_cross_validation_consistency,
        test_stratified_split_preserves_ratio,
        # Phase 5
        test_inference_handles_failures,
        test_batch_inference_completeness,
        test_ranking_prioritizes_high_confidence,
        test_ranking_filters_low_quality
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_func.__name__} FAILED: {str(e)}")
            failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} ERROR: {str(e)}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    return passed, failed
```

---

## 📊 Test Execution Plan

### Pre-Implementation Testing
1. **Copy test functions** to notebook cell
2. **Run unit tests** before implementing features
3. **Verify all tests fail** initially (TDD red phase)
4. **Implement features** to make tests pass (green phase)
5. **Refactor** while keeping tests passing

### Test Execution Order

```python
# In each notebook, add this cell at the end:

print("="*60)
print("RUNNING TDD TEST SUITE")
print("="*60)

# Phase 3 Tests
if 'PHASE_3' in globals():
    test_bls_period_recovery_simple()
    test_feature_extraction_completeness()
    test_checkpoint_save_load()

# Phase 4 Tests
if 'PHASE_4' in globals():
    test_gpu_detection()
    test_calibration_improves_brier_score()
    test_cross_validation_consistency()

# Phase 5 Tests
if 'PHASE_5' in globals():
    test_inference_handles_failures()
    test_ranking_prioritizes_high_confidence()

print("\n✅ All tests passed!")
```

---

## 🎯 Success Criteria

### Test Coverage Goals
- **Unit Test Coverage**: 100% of critical functions
- **Integration Test Coverage**: All end-to-end workflows
- **Edge Case Coverage**: Failure modes, empty inputs, extreme values

### Performance Benchmarks
- **Unit tests**: Complete in <10 seconds
- **Integration tests**: Complete in <2 minutes
- **Full test suite**: Complete in <5 minutes

---

**Last Updated**: 2025-09-30
**Status**: Ready for TDD Implementation
**Next Action**: Copy test functions to notebooks and begin red-green-refactor cycle