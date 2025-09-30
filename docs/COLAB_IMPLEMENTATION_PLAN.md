# 🚀 Colab Implementation Plan - Complete Exoplanet Detection Pipeline

**Document Version**: 1.0
**Date**: 2025-09-30
**Status**: Ready for Implementation
**Target Dataset Size**: 11,979 samples (5,944 positive, 6,035 negative)

---

## 📋 Executive Summary

This plan outlines the implementation of three production-ready Google Colab notebooks to process the **full 11,979-sample dataset** using GPU acceleration, TDD methodology, and Colab-optimized workflows.

### Current Status
- ✅ **Phase 1-2**: Data downloaded (`supervised_dataset.csv` with 11,979 samples)
- 📋 **Phase 3**: BLS/TLS feature extraction (THIS PLAN)
- 📋 **Phase 4**: XGBoost training with GPU
- 📋 **Phase 5**: Inference pipeline

### Key Objectives
1. Extract BLS/TLS features from all 11,979 light curves
2. Train calibrated XGBoost model with GPU acceleration
3. Deploy one-click inference pipeline for new targets
4. Maintain full reproducibility and Colab compatibility

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     COLAB EXECUTION ENVIRONMENT                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  02_bls_baseline.ipynb (Phase 3)                              │ │
│  │  ─────────────────────────────────────────────────────────────│ │
│  │  Input:  supervised_dataset.csv (11,979 samples)              │ │
│  │  Process: BLS/TLS feature extraction                           │ │
│  │  Output:  bls_tls_features.csv                                 │ │
│  │  GPU:     N/A (CPU-intensive, parallelized)                   │ │
│  │  Time:    ~6-8 hours (batch processing)                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  03_injection_train.ipynb (Phase 4)                           │ │
│  │  ─────────────────────────────────────────────────────────────│ │
│  │  Input:  bls_tls_features.csv                                  │ │
│  │  Process: XGBoost training + calibration                       │ │
│  │  Output:  xgboost_model.joblib, calibrator.joblib             │ │
│  │  GPU:     T4/L4 (tree_method='gpu_hist')                      │ │
│  │  Time:    ~30-60 minutes                                       │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                              ↓                                        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  04_newdata_inference.ipynb (Phase 5)                         │ │
│  │  ─────────────────────────────────────────────────────────────│ │
│  │  Input:  TIC IDs (user input)                                  │ │
│  │  Process: MAST → BLS/TLS → Model → Probabilities             │ │
│  │  Output:  candidates_YYYYMMDD.csv                             │ │
│  │  GPU:     N/A (inference is fast)                             │ │
│  │  Time:    ~5-10 min per target                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Notebook 1: BLS/TLS Feature Extraction (02_bls_baseline.ipynb)

### 🎯 Objective
Extract 14 BLS/TLS features from all 11,979 light curves with robust error handling and progress tracking.

### 📐 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Feature Extraction Pipeline                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  [Load Data] → [Download LCs] → [BLS/TLS] → [Features] → [Save] │
│       ↓              ↓              ↓            ↓           ↓    │
│   11,979         MAST API      run_bls()    extract_      CSV   │
│   samples        Lightkurve    run_tls()    features()          │
│                                                                    │
│  Parallelization Strategy:                                        │
│  ├─ Batch size: 50 targets                                       │
│  ├─ Checkpoint: Every 100 samples                                │
│  ├─ Fallback: Skip failed targets (log errors)                  │
│  └─ Resume: From last checkpoint                                 │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 🔧 Implementation Details

#### Cell Structure
```python
# Cell 1: Colab Setup
!pip install -q numpy==1.26.4 scipy'<1.13' astropy lightkurve transitleastsquares wotan

# Cell 2: Import & Configure
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import pandas as pd
import numpy as np
import lightkurve as lk
from transitleastsquares import transitleastsquares
from multiprocessing import Pool, cpu_count
import json
from datetime import datetime

# Cell 3: Load Dataset
df = pd.read_csv('data/supervised_dataset.csv')
print(f"📊 Total samples: {len(df)}")

# Cell 4: Feature Extraction Function (from app/bls_features.py)
# [Copy entire bls_features.py module]

# Cell 5: Batch Processing with Checkpoints
BATCH_SIZE = 50
CHECKPOINT_INTERVAL = 100
OUTPUT_DIR = Path('data/features')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cell 6: Main Extraction Loop
# [Parallel processing with error handling]

# Cell 7: Validation & Save
# [Check feature quality, save final CSV]

# Cell 8: GitHub Push (Optional)
# [Push results back to GitHub]
```

#### Key Functions

**1. Robust Light Curve Download**
```python
def download_lightcurve_safe(tic_id: str, max_retries: int = 3) -> Optional[lk.LightCurve]:
    """
    Download TESS light curve with fallback strategies

    Strategies:
    1. Try SPOC (Science Processing Operations Center)
    2. Try QLP (Quick-Look Pipeline)
    3. Try TESS-SPOC 120s cadence
    4. Return None if all fail
    """
    for retry in range(max_retries):
        try:
            # Strategy 1: SPOC 2-minute cadence
            search = lk.search_lightcurve(tic_id, author='SPOC', mission='TESS')
            if len(search) > 0:
                lc = search[0].download()
                return lc.remove_nans().normalize()

            # Strategy 2: QLP
            search = lk.search_lightcurve(tic_id, author='QLP', mission='TESS')
            if len(search) > 0:
                lc = search[0].download()
                return lc.remove_nans().normalize()

        except Exception as e:
            if retry == max_retries - 1:
                print(f"❌ Failed to download {tic_id}: {str(e)}")
                return None
            time.sleep(2 ** retry)  # Exponential backoff

    return None
```

**2. Checkpoint System**
```python
def save_checkpoint(features_df: pd.DataFrame, checkpoint_id: int):
    """Save intermediate results"""
    checkpoint_path = OUTPUT_DIR / f'checkpoint_{checkpoint_id:04d}.parquet'
    features_df.to_parquet(checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")

def load_latest_checkpoint() -> Tuple[pd.DataFrame, int]:
    """Resume from last checkpoint"""
    checkpoints = sorted(OUTPUT_DIR.glob('checkpoint_*.parquet'))
    if not checkpoints:
        return pd.DataFrame(), 0

    latest = checkpoints[-1]
    df = pd.read_parquet(latest)
    checkpoint_id = int(latest.stem.split('_')[1])
    print(f"🔄 Resuming from checkpoint {checkpoint_id}")
    return df, checkpoint_id
```

**3. Parallel Batch Processing**
```python
def process_batch(batch_df: pd.DataFrame) -> pd.DataFrame:
    """Process a batch of targets in parallel"""
    features_list = []

    for idx, row in batch_df.iterrows():
        try:
            # Download light curve
            lc = download_lightcurve_safe(row['tic_id'])
            if lc is None:
                continue

            # Extract BLS features
            bls_result = run_bls(lc.time.value, lc.flux.value)

            # Extract TLS features (if BLS found signal)
            if bls_result['snr'] > 5.0:
                tls_result = run_tls_safe(lc)
            else:
                tls_result = {}

            # Combine features
            features = extract_features(
                lc.time.value,
                lc.flux.value,
                bls_result,
                tls_result,
                compute_advanced=True
            )
            features['tic_id'] = row['tic_id']
            features['label'] = row['label']
            features_list.append(features)

        except Exception as e:
            print(f"⚠️  Error processing {row['tic_id']}: {str(e)}")
            continue

    return pd.DataFrame(features_list)
```

### ⏱️ Time & Resource Estimates

| Metric | Value | Notes |
|--------|-------|-------|
| **Total Targets** | 11,979 | Full dataset |
| **Success Rate** | ~85% | Assuming 15% download failures |
| **Processed Targets** | ~10,182 | Expected successful extractions |
| **Time per Target** | ~2-3 sec | BLS + basic features |
| **Total Time (Sequential)** | ~6.6 hrs | 11,979 × 2 sec |
| **Total Time (8-core)** | ~50 min | Parallelized |
| **With TLS (30%)** | ~2 hrs | TLS on high-SNR targets |
| **Checkpoints** | Every 100 | ~120 checkpoints total |
| **Output Size** | ~5 MB | CSV file |

### 🧪 TDD Test Strategy

**Test Cases:**
```python
def test_bls_extraction():
    """Test BLS on synthetic transit"""
    # Generate synthetic data
    time = np.linspace(0, 27, 1000)
    flux = np.ones(1000)

    # Inject box transit
    period, depth, duration = 3.5, 0.01, 0.1
    flux_injected = inject_box_transit(time, flux, period, depth, duration, t0=0)

    # Run BLS
    bls_result = run_bls(time, flux_injected)

    # Assertions
    assert abs(bls_result['period'] - period) < 0.1, "Period recovery failed"
    assert abs(bls_result['depth'] - depth) < 0.005, "Depth recovery failed"
    assert bls_result['snr'] > 10, "SNR too low"

def test_feature_extraction():
    """Test feature dictionary completeness"""
    time = np.linspace(0, 27, 1000)
    flux = np.random.normal(1, 0.001, 1000)
    bls_result = {'period': 3.5, 'depth': 0.01, 'duration': 0.1, 'snr': 15, 't0': 0}

    features = extract_features(time, flux, bls_result)

    # Check all required features exist
    required_features = [
        'bls_period', 'bls_depth', 'bls_duration', 'bls_snr',
        'duration_over_period', 'depth_snr_ratio',
        'odd_even_depth_diff', 'transit_symmetry',
        'flux_std', 'flux_mad', 'flux_skew', 'flux_kurtosis',
        'periodicity_strength'
    ]

    for feat in required_features:
        assert feat in features, f"Missing feature: {feat}"
        assert not np.isnan(features[feat]), f"NaN feature: {feat}"

def test_checkpoint_system():
    """Test checkpoint save/load"""
    # Create dummy features
    features_df = pd.DataFrame({'tic_id': ['TIC1', 'TIC2'], 'feature1': [1, 2]})

    # Save checkpoint
    save_checkpoint(features_df, 1)

    # Load checkpoint
    loaded_df, checkpoint_id = load_latest_checkpoint()

    assert len(loaded_df) == 2, "Checkpoint load failed"
    assert checkpoint_id == 1, "Checkpoint ID mismatch"
```

### 📦 Deliverables

1. **bls_tls_features.csv** (primary output)
   - Columns: `tic_id`, `label`, 14 feature columns
   - Rows: ~10,182 (85% success rate)

2. **feature_extraction_report.json**
   ```json
   {
     "total_targets": 11979,
     "successful": 10182,
     "failed": 1797,
     "failure_reasons": {
       "no_data": 1200,
       "download_error": 400,
       "processing_error": 197
     },
     "execution_time_minutes": 120,
     "checkpoints": 120
   }
   ```

3. **Checkpoint files** (temporary)
   - `checkpoint_0000.parquet` to `checkpoint_0119.parquet`

---

## 🤖 Notebook 2: XGBoost Training (03_injection_train.ipynb)

### 🎯 Objective
Train GPU-accelerated XGBoost model with isotonic calibration and comprehensive evaluation.

### 📐 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Training Pipeline (GPU)                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  [Load Features] → [Preprocess] → [Train XGBoost] → [Calibrate] │
│         ↓               ↓               ↓                ↓        │
│   10,182 samples   StandardScaler   GPU Accel.    Isotonic      │
│   14 features      Handle NaN       tree_method   Regression    │
│   5:5 ratio        SMOTE (opt)      gpu_hist                    │
│                                         ↓                         │
│                              ┌─────────────────┐                 │
│                              │ Cross-Validation│                 │
│                              │  StratifiedKFold│                 │
│                              │     5 folds     │                 │
│                              └─────────────────┘                 │
│                                         ↓                         │
│                              [Model Evaluation]                   │
│                              - ROC-AUC, PR-AUC                   │
│                              - Calibration curve                 │
│                              - Feature importance                │
│                                         ↓                         │
│                              [Save Model + Metadata]             │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 🔧 Implementation Details

#### Cell Structure
```python
# Cell 1: GPU Setup
!pip install -q xgboost scikit-learn imbalanced-learn

import torch
print(f"🚀 GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")

# Cell 2: Load Features
features_df = pd.read_csv('data/bls_tls_features.csv')
print(f"📊 Features shape: {features_df.shape}")

# Cell 3: Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Cell 4: XGBoost Training (GPU)
import xgboost as xgb

params = {
    'tree_method': 'gpu_hist',  # GPU acceleration
    'gpu_id': 0,
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'early_stopping_rounds': 50,
    'eval_metric': ['auc', 'logloss'],
    'scale_pos_weight': 1.0  # Balanced dataset
}

# Cell 5: Cross-Validation
from sklearn.model_selection import StratifiedKFold

# Cell 6: Calibration
from sklearn.calibration import CalibratedClassifierCV

# Cell 7: Evaluation & Visualization
# ROC curve, PR curve, calibration curve

# Cell 8: Save Model
import joblib
joblib.dump(model, 'model/xgboost_calibrated.joblib')
```

#### Key Functions

**1. GPU-Optimized Training**
```python
def train_xgboost_gpu(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Dict[str, Any]
) -> xgb.XGBClassifier:
    """
    Train XGBoost with GPU acceleration and early stopping
    """
    # Create DMatrix for GPU
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # GPU-specific parameters
    gpu_params = {
        'tree_method': 'gpu_hist',
        'gpu_id': 0,
        'predictor': 'gpu_predictor',
        **params
    }

    # Train with evaluation
    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        gpu_params,
        dtrain,
        num_boost_round=params.get('n_estimators', 500),
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50
    )

    return model
```

**2. Isotonic Calibration**
```python
def calibrate_model(
    base_model: xgb.XGBClassifier,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    method: str = 'isotonic'
) -> CalibratedClassifierCV:
    """
    Calibrate model probabilities using isotonic regression

    Isotonic regression is ideal for exoplanet detection because:
    - Preserves ranking order
    - Handles non-linear probability distortions
    - Works well with small datasets
    """
    calibrated = CalibratedClassifierCV(
        base_model,
        method=method,
        cv='prefit'
    )
    calibrated.fit(X_cal, y_cal)

    return calibrated
```

**3. Comprehensive Evaluation**
```python
def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path
) -> Dict[str, float]:
    """
    Comprehensive model evaluation with visualization
    """
    # Predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Metrics
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        brier_score_loss, log_loss
    )

    metrics = {
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba),
        'brier_score': brier_score_loss(y_test, y_proba),
        'log_loss': log_loss(y_test, y_proba)
    }

    # ECE (Expected Calibration Error)
    metrics['ece'] = calculate_ece(y_test, y_proba, n_bins=10)

    # Visualizations
    plot_roc_curve(y_test, y_proba, output_dir)
    plot_pr_curve(y_test, y_proba, output_dir)
    plot_calibration_curve(y_test, y_proba, output_dir)
    plot_feature_importance(model, output_dir)

    return metrics
```

### ⏱️ Time & Resource Estimates

| Metric | Value | Notes |
|--------|-------|-------|
| **Training Samples** | ~8,145 | 80% split |
| **Validation Samples** | ~2,037 | 20% split |
| **Features** | 14 | From BLS/TLS extraction |
| **GPU Type** | T4/L4 | Colab Pro recommended |
| **Training Time (GPU)** | ~10-15 min | 500 trees, early stopping |
| **Training Time (CPU)** | ~45-60 min | Fallback option |
| **Calibration Time** | ~2 min | Isotonic regression |
| **Total Time** | ~20-30 min | Including evaluation |
| **Model Size** | ~5 MB | XGBoost + calibrator |

### 🧪 TDD Test Strategy

**Test Cases:**
```python
def test_xgboost_gpu_availability():
    """Verify GPU acceleration is enabled"""
    params = {'tree_method': 'gpu_hist', 'gpu_id': 0}

    try:
        dtrain = xgb.DMatrix(np.random.rand(100, 10), label=np.random.randint(0, 2, 100))
        model = xgb.train(params, dtrain, num_boost_round=10)
        assert True, "GPU training successful"
    except Exception as e:
        print(f"⚠️  GPU training failed, falling back to CPU: {e}")
        params['tree_method'] = 'hist'

def test_calibration_quality():
    """Test calibration improves Brier score"""
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train base model
    base_model = xgb.XGBClassifier(n_estimators=50)
    base_model.fit(X_train, y_train)
    y_proba_base = base_model.predict_proba(X_test)[:, 1]

    # Calibrate
    calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
    calibrated.fit(X_train, y_train)
    y_proba_cal = calibrated.predict_proba(X_test)[:, 1]

    # Compare Brier scores
    brier_base = brier_score_loss(y_test, y_proba_base)
    brier_cal = brier_score_loss(y_test, y_proba_cal)

    assert brier_cal <= brier_base, "Calibration should improve Brier score"

def test_cross_validation():
    """Test 5-fold CV produces consistent results"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X, y = make_classification(n_samples=1000, n_features=10)

    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(n_estimators=50)
        model.fit(X_train, y_train)
        score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        scores.append(score)

    assert np.std(scores) < 0.1, "CV scores should be consistent"
```

### 📦 Deliverables

1. **xgboost_calibrated.joblib** (primary model)
2. **scaler.joblib** (feature scaler)
3. **training_report.json**
   ```json
   {
     "model_type": "XGBoost",
     "gpu_accelerated": true,
     "training_samples": 8145,
     "validation_samples": 2037,
     "features": 14,
     "metrics": {
       "roc_auc": 0.952,
       "pr_auc": 0.948,
       "brier_score": 0.082,
       "ece": 0.034
     },
     "hyperparameters": {
       "max_depth": 8,
       "learning_rate": 0.05,
       "n_estimators": 347,
       "early_stopped": true
     },
     "training_time_minutes": 15,
     "feature_importance": {
       "bls_snr": 0.185,
       "bls_depth": 0.142,
       "odd_even_depth_diff": 0.118
     }
   }
   ```

4. **Visualization outputs**
   - `roc_curve.png`
   - `pr_curve.png`
   - `calibration_curve.png`
   - `feature_importance.png`

---

## 🔮 Notebook 3: Inference Pipeline (04_newdata_inference.ipynb)

### 🎯 Objective
One-click inference pipeline for new TIC targets with automatic candidate ranking.

### 📐 Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Inference Pipeline                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  [User Input] → [Download LC] → [BLS/TLS] → [Model] → [Results] │
│       ↓              ↓              ↓           ↓          ↓      │
│   TIC IDs        MAST API      Features    XGBoost    Ranked    │
│   (list)         Lightkurve    Extraction  Calibrated  CSV      │
│                                                                    │
│  Batch Processing:                                                │
│  ├─ Input: List of TIC IDs                                       │
│  ├─ Parallel downloads (4 concurrent)                            │
│  ├─ Feature extraction (reuse functions)                         │
│  ├─ Model prediction (vectorized)                                │
│  └─ Output: Ranked candidates CSV                                │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 🔧 Implementation Details

#### Cell Structure
```python
# Cell 1: Setup
!pip install -q lightkurve transitleastsquares joblib

# Cell 2: Load Model
import joblib
model = joblib.load('model/xgboost_calibrated.joblib')
scaler = joblib.load('model/scaler.joblib')

# Cell 3: User Input
TIC_IDS = [
    'TIC 25155310',
    'TIC 307210830',
    'TIC 141527766'
]

# Cell 4: Inference Function (from app/infer.py)
# [Copy inference functions]

# Cell 5: Run Inference
results_df = batch_inference(TIC_IDS, model, scaler)

# Cell 6: Rank & Display
results_df = results_df.sort_values('planet_probability', ascending=False)
print(results_df[['tic_id', 'planet_probability', 'bls_period', 'bls_depth']])

# Cell 7: Save Results
results_df.to_csv(f'outputs/candidates_{datetime.now():%Y%m%d}.csv')
```

#### Key Functions

**1. End-to-End Inference**
```python
def infer_single_target(
    tic_id: str,
    model: Any,
    scaler: StandardScaler
) -> Dict[str, Any]:
    """
    Complete inference pipeline for a single target

    Steps:
    1. Download light curve from MAST
    2. Preprocess (remove NaNs, normalize)
    3. Extract BLS/TLS features
    4. Scale features
    5. Predict with calibrated model
    6. Return result dictionary
    """
    try:
        # Download
        lc = download_lightcurve_safe(tic_id)
        if lc is None:
            return {'tic_id': tic_id, 'error': 'download_failed'}

        # Feature extraction
        bls_result = run_bls(lc.time.value, lc.flux.value)
        features = extract_features(lc.time.value, lc.flux.value, bls_result)

        # Prepare feature vector (ensure correct order)
        feature_vector = np.array([features[feat] for feat in model.feature_names_])
        feature_vector_scaled = scaler.transform([feature_vector])

        # Predict
        proba = model.predict_proba(feature_vector_scaled)[0, 1]

        return {
            'tic_id': tic_id,
            'planet_probability': proba,
            'bls_period': bls_result['period'],
            'bls_depth': bls_result['depth'],
            'bls_snr': bls_result['snr'],
            'status': 'success'
        }

    except Exception as e:
        return {
            'tic_id': tic_id,
            'error': str(e),
            'status': 'failed'
        }
```

**2. Batch Inference**
```python
def batch_inference(
    tic_ids: List[str],
    model: Any,
    scaler: StandardScaler,
    n_workers: int = 4
) -> pd.DataFrame:
    """
    Process multiple targets in parallel
    """
    from concurrent.futures import ThreadPoolExecutor

    print(f"🚀 Processing {len(tic_ids)} targets with {n_workers} workers...")

    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(infer_single_target, tic_id, model, scaler)
            for tic_id in tic_ids
        ]

        for i, future in enumerate(futures, 1):
            result = future.result()
            results.append(result)

            if i % 10 == 0:
                print(f"   Processed {i}/{len(tic_ids)}...")

    return pd.DataFrame(results)
```

**3. Candidate Ranking**
```python
def rank_candidates(
    results_df: pd.DataFrame,
    min_probability: float = 0.7,
    min_snr: float = 7.0
) -> pd.DataFrame:
    """
    Rank candidates by multiple criteria

    Ranking factors:
    1. Planet probability (primary)
    2. BLS SNR (confidence)
    3. Period range (habitable zone preference)
    4. Depth (detectability)
    """
    # Filter by minimum thresholds
    candidates = results_df[
        (results_df['planet_probability'] >= min_probability) &
        (results_df['bls_snr'] >= min_snr) &
        (results_df['status'] == 'success')
    ].copy()

    # Calculate composite score
    candidates['score'] = (
        0.60 * candidates['planet_probability'] +
        0.25 * (candidates['bls_snr'] / 50.0).clip(0, 1) +
        0.15 * (1 - abs(candidates['bls_period'] - 10) / 10).clip(0, 1)
    )

    # Sort by score
    candidates = candidates.sort_values('score', ascending=False)

    return candidates
```

### ⏱️ Time & Resource Estimates

| Metric | Value | Notes |
|--------|-------|-------|
| **Time per Target** | ~5-10 min | Download + BLS + inference |
| **Batch of 10** | ~15 min | Parallel downloads (4 workers) |
| **Batch of 100** | ~2 hours | Checkpointed |
| **Model Load Time** | <1 sec | Joblib deserialization |
| **Inference Time** | <0.1 sec | Per target (after features) |
| **Output Size** | ~100 KB | Per 100 targets |

### 🧪 TDD Test Strategy

**Test Cases:**
```python
def test_inference_pipeline():
    """Test end-to-end inference on known planet"""
    known_planet_tic = 'TIC 25155310'  # TOI 270 b

    result = infer_single_target(known_planet_tic, model, scaler)

    assert result['status'] == 'success', "Inference failed"
    assert result['planet_probability'] > 0.8, "Should detect known planet"

def test_batch_inference():
    """Test batch processing"""
    tic_ids = ['TIC 25155310', 'TIC 307210830']

    results_df = batch_inference(tic_ids, model, scaler)

    assert len(results_df) == len(tic_ids), "Missing results"
    assert 'planet_probability' in results_df.columns, "Missing probability"

def test_ranking_system():
    """Test candidate ranking logic"""
    results_df = pd.DataFrame({
        'tic_id': ['TIC1', 'TIC2', 'TIC3'],
        'planet_probability': [0.95, 0.85, 0.75],
        'bls_snr': [30, 20, 10],
        'bls_period': [5, 15, 25],
        'status': ['success', 'success', 'success']
    })

    ranked = rank_candidates(results_df, min_probability=0.7)

    assert len(ranked) == 3, "Filtering error"
    assert ranked.iloc[0]['tic_id'] == 'TIC1', "Ranking error"
```

### 📦 Deliverables

1. **candidates_YYYYMMDD.csv** (primary output)
   ```csv
   tic_id,planet_probability,bls_period,bls_depth,bls_snr,score,rank
   TIC 25155310,0.952,3.36,0.0082,28.5,0.887,1
   TIC 307210830,0.885,5.72,0.0045,22.1,0.792,2
   ```

2. **inference_report.json**
   ```json
   {
     "execution_timestamp": "2025-09-30T10:30:00",
     "total_targets": 150,
     "successful": 142,
     "failed": 8,
     "high_confidence_candidates": 12,
     "medium_confidence_candidates": 28,
     "execution_time_minutes": 45
   }
   ```

---

## 🔄 Workflow Integration

### Complete Pipeline Execution Order

```
Day 1: Feature Extraction
├─ Step 1: Open 02_bls_baseline.ipynb in Colab
├─ Step 2: Run all cells (6-8 hours with checkpoints)
├─ Step 3: Download bls_tls_features.csv to local
└─ Step 4: Optionally push to GitHub

Day 2: Model Training
├─ Step 1: Open 03_injection_train.ipynb in Colab
├─ Step 2: Enable GPU (Runtime → Change runtime type → T4 GPU)
├─ Step 3: Run all cells (~30 minutes)
├─ Step 4: Download model files (xgboost_calibrated.joblib)
└─ Step 5: Review training metrics

Day 3: Production Inference
├─ Step 1: Open 04_newdata_inference.ipynb
├─ Step 2: Upload model files from Day 2
├─ Step 3: Input target TIC IDs
├─ Step 4: Run inference (~5-10 min per target)
└─ Step 5: Download ranked candidates CSV
```

### Checkpoint & Resume Strategy

**For 02_bls_baseline.ipynb:**
```python
# At start of notebook
if Path('data/features/checkpoint_latest.parquet').exists():
    features_df, last_idx = load_latest_checkpoint()
    df = df.iloc[last_idx:]  # Resume from last checkpoint
else:
    features_df = pd.DataFrame()
    last_idx = 0

# During processing
for batch_id in range(n_batches):
    batch_features = process_batch(...)
    features_df = pd.concat([features_df, batch_features])

    if (batch_id + 1) % 10 == 0:  # Every 10 batches
        save_checkpoint(features_df, batch_id)
```

---

## 📊 Performance Optimization

### GPU Acceleration Opportunities

| Component | GPU Support | Speedup | Implementation |
|-----------|-------------|---------|----------------|
| BLS Search | ❌ (CPU) | 1x | Use multiprocessing |
| TLS Search | ❌ (CPU) | 1x | Skip low-SNR targets |
| XGBoost Training | ✅ (T4/L4) | 3-5x | `tree_method='gpu_hist'` |
| XGBoost Inference | ✅ (T4/L4) | 2-3x | `predictor='gpu_predictor'` |
| Feature Scaling | ❌ (CPU) | 1x | Vectorized NumPy |

### Memory Management

**Colab Constraints:**
- Standard: 12.7 GB RAM
- High-RAM: 25.5 GB RAM (Colab Pro)
- GPU Memory: 15 GB (T4), 22 GB (L4)

**Optimization Strategies:**
1. **Batch Processing**: Process 50 targets at a time
2. **Checkpointing**: Save every 100 samples
3. **Garbage Collection**: Force GC after each batch
4. **Lazy Loading**: Load data in chunks with `pd.read_csv(chunksize=1000)`

```python
import gc

for batch_id, batch_df in enumerate(pd.read_csv('data/supervised_dataset.csv', chunksize=50)):
    # Process batch
    features = process_batch(batch_df)

    # Save checkpoint
    if batch_id % 20 == 0:
        save_checkpoint(features, batch_id)
        gc.collect()  # Force garbage collection
```

---

## 🛡️ Error Handling & Recovery

### Common Failure Modes

| Error Type | Frequency | Mitigation |
|------------|-----------|------------|
| MAST API Timeout | ~5% | Retry with exponential backoff |
| No Data Available | ~10% | Skip target, log error |
| Lightkurve Error | ~3% | Fallback to different mission/author |
| BLS Failure | ~2% | Return NaN features, continue |
| Memory Error | <1% | Reduce batch size, checkpoint |
| Colab Timeout | Runtime | Save checkpoints every 100 samples |

### Error Handling Pattern

```python
def robust_processing(target: str, max_retries: int = 3) -> Optional[Dict]:
    """Process with retry logic and fallback"""
    for attempt in range(max_retries):
        try:
            # Attempt processing
            result = process_target(target)
            return result

        except requests.Timeout:
            # Network issue - retry with backoff
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            else:
                log_error(target, 'timeout')
                return None

        except ValueError as e:
            # Data quality issue - skip
            log_error(target, f'data_quality: {str(e)}')
            return None

        except Exception as e:
            # Unexpected error - log and skip
            log_error(target, f'unexpected: {str(e)}')
            return None
```

---

## 📈 Success Metrics

### Completion Criteria (Phase 3)
- ✅ Processed ≥85% of 11,979 targets (≥10,182 successful)
- ✅ Generated `bls_tls_features.csv` with 14 features
- ✅ No NaN values in critical features (period, depth, SNR)
- ✅ Execution time <8 hours with checkpoints

### Completion Criteria (Phase 4)
- ✅ XGBoost model trained with GPU acceleration
- ✅ ROC-AUC ≥0.92 on test set
- ✅ PR-AUC ≥0.90 on test set
- ✅ ECE (calibration error) <0.05
- ✅ Training time <30 minutes on T4 GPU

### Completion Criteria (Phase 5)
- ✅ Inference pipeline runs on new TIC IDs
- ✅ Correctly ranks known planets in top 10%
- ✅ Inference time <10 min per target
- ✅ Produces CSV with ranked candidates

---

## 🚀 Quick Start Checklist

### Prerequisites
- [ ] Google account with Colab access
- [ ] GitHub repository with `supervised_dataset.csv`
- [ ] Colab Pro (recommended for GPU and longer runtimes)

### Execution Steps

**Phase 3: Feature Extraction**
```bash
# 1. Open in Colab
https://colab.research.google.com/github/YOUR_USERNAME/exoplanet-starter/blob/main/notebooks/02_bls_baseline.ipynb

# 2. Run sequentially
Cell 1-3: Setup (5 min)
Cell 4: Feature extraction (6-8 hours)
Cell 5: Validation & save (2 min)

# 3. Download results
bls_tls_features.csv (5 MB)
```

**Phase 4: Training**
```bash
# 1. Open in Colab
https://colab.research.google.com/github/YOUR_USERNAME/exoplanet-starter/blob/main/notebooks/03_injection_train.ipynb

# 2. Enable GPU
Runtime → Change runtime type → Hardware accelerator: T4 GPU

# 3. Run all cells (30 min)

# 4. Download model
xgboost_calibrated.joblib (5 MB)
scaler.joblib (1 KB)
```

**Phase 5: Inference**
```bash
# 1. Open in Colab
https://colab.research.google.com/github/YOUR_USERNAME/exoplanet-starter/blob/main/notebooks/04_newdata_inference.ipynb

# 2. Upload model files (from Phase 4)

# 3. Input TIC IDs
TIC_IDS = ['TIC 25155310', 'TIC 307210830', ...]

# 4. Run inference (5-10 min per target)

# 5. Download candidates
candidates_20250930.csv
```

---

## 📚 References & Documentation

### NASA Data Sources
- [NASA Exoplanet Archive TAP](https://exoplanetarchive.ipac.caltech.edu/TAP)
- [MAST Archive](https://mast.stsci.edu/)
- [TESS Mission](https://tess.mit.edu/)

### Key Libraries
- [Lightkurve](https://docs.lightkurve.org/) - Light curve analysis
- [TransitLeastSquares](https://github.com/hippke/tls) - Transit search
- [XGBoost](https://xgboost.readthedocs.io/) - Machine learning
- [scikit-learn](https://scikit-learn.org/) - ML utilities

### Related Documentation
- `PROJECT_MEMORY.md` - Project history and solutions
- `CLAUDE.md` - Development guidelines
- `DATASETS.md` - Data source documentation
- `COLAB_TROUBLESHOOTING.md` - Known issues and fixes

---

## 🎯 Next Actions

1. **Review this plan** with stakeholders
2. **Execute Phase 3** (02_bls_baseline.ipynb)
3. **Monitor progress** via checkpoints
4. **Validate features** before proceeding to Phase 4
5. **Document any issues** in `PROJECT_MEMORY.md`

---

**Document Status**: ✅ Ready for Implementation
**Last Updated**: 2025-09-30
**Next Review**: After Phase 3 completion
**Contact**: See `CLAUDE.md` for development guidance