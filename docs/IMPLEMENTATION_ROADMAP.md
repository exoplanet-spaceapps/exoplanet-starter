# 🚀 Exoplanet Detection Project - Implementation Roadmap
**Date:** 2025-09-30
**Author:** Development Team
**Status:** Ready for Implementation

## 📋 Executive Summary

This roadmap provides a comprehensive plan to transform the exoplanet detection project into a production-ready, 2025-standard ML pipeline. All improvements follow Test-Driven Development (TDD) principles and industry best practices.

## ✅ Phase 0: COMPLETED (2025-09-30)

### Achievements:
- ✅ Fixed all notebook syntax errors (5 notebooks)
- ✅ Implemented UTF-8 encoding support
- ✅ Refactored data loading (81% code reduction)
- ✅ Created comprehensive TDD test suite (14/14 tests passing)
- ✅ Verified data loading with 11,979 samples
- ✅ Pushed all fixes to GitHub

### Commit: `fd48b09`
**Message:** "feat: comprehensive notebook fixes and TDD implementation"

---

## 🎯 Phase 1: Critical Infrastructure (Week 1)

### Priority: IMMEDIATE
**Goal:** Make all notebooks production-ready with proper error handling and reproducibility

### 1.1 Reproducibility Framework
**Create:** `src/utils/reproducibility.py`

```python
"""Reproducibility utilities - 2025 standards."""
import random
import numpy as np
import os

GLOBAL_RANDOM_STATE = 42

def set_random_seeds(seed: int = GLOBAL_RANDOM_STATE) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ Random seeds set to {seed}")
```

**Apply to:**
- 03_injection_train.ipynb: Add to Cell 2
- All data splitting operations
- All model training
- Cross-validation

### 1.2 Logging Infrastructure
**Create:** `src/utils/logging_config.py`

```python
"""Centralized logging configuration."""
import logging
from pathlib import Path

def setup_logger(name: str, log_file: Path = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(console)

    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(console.formatter)
        logger.addHandler(file_handler)

    return logger
```

**Apply to ALL notebooks:**
```python
# Cell 2: Setup logging
import sys
sys.path.insert(0, '../src')
from utils.logging_config import setup_logger
from utils.reproducibility import set_random_seeds

logger = setup_logger(__name__, Path('../logs/notebook_XX.log'))
set_random_seeds(42)
logger.info("Starting notebook execution")
```

### 1.3 Data Verification
**Add to 02_bls_baseline.ipynb Cell 7:**

```python
# Verify data availability before proceeding
if len(sample_targets) == 0:
    raise ValueError("No samples loaded! Run 01_tap_download.ipynb first.")

logger.info(f"Loaded {len(sample_targets)} targets for analysis")
logger.info(f"Positive: {(sample_targets['label']==1).sum()}, "
           f"Negative: {(sample_targets['label']==0).sum()}")
```

### Deliverables:
- [ ] `src/utils/reproducibility.py` created and tested
- [ ] `src/utils/logging_config.py` created and tested
- [ ] All notebooks updated with logging and seed setting
- [ ] Tests pass: `pytest tests/test_reproducibility.py -v`

---

## ⚡ Phase 2: GPU Optimization (Week 1-2)

### Priority: HIGH
**Goal:** Enable GPU acceleration for XGBoost training and inference

### 2.1 XGBoost GPU Support - 03_injection_train.ipynb
**Reference:** https://xgboost.readthedocs.io/en/stable/gpu/

**Add Cell 3: GPU Detection**
```python
# Detect GPU availability
import subprocess

def detect_gpu():
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ GPU detected - enabling CUDA acceleration")
            print(result.stdout.split('\n')[8])  # GPU info line
            return True
    except FileNotFoundError:
        pass
    print("⚠️  No GPU detected - using CPU")
    return False

GPU_AVAILABLE = detect_gpu()
```

**Update Training Cell (current location: Cell ~20):**
```python
# XGBoost with GPU support (XGBoost 2.x)
xgb_params = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.05,
    'random_state': 42,  # REPRODUCIBILITY
    'tree_method': 'hist',  # Fast histogram method
    'device': 'cuda' if GPU_AVAILABLE else 'cpu',  # XGBoost 2.x
    'predictor': 'gpu_predictor' if GPU_AVAILABLE else 'cpu_predictor',
    'eval_metric': 'logloss',
    'early_stopping_rounds': 10
}

from xgboost import XGBClassifier
model = XGBClassifier(**xgb_params)

logger.info(f"Training on {xgb_params['device'].upper()}")
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
```

### 2.2 GPU Inference - 04_newdata_inference.ipynb
**Add after model loading:**
```python
# Enable GPU for inference if available
if GPU_AVAILABLE:
    loaded_model.set_params(device='cuda', predictor='gpu_predictor')
    logger.info("Using GPU for inference")
```

### 2.3 Colab GPU Setup
**Add to ALL notebooks Cell 1:**
```python
# GPU Environment Setup
!pip install -q xgboost[gpu] >/dev/null 2>&1

import subprocess
import sys

GPU_AVAILABLE = False
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True)
    GPU_AVAILABLE = result.returncode == 0
    if GPU_AVAILABLE:
        print("✅ GPU Available - XGBoost will use CUDA")
except:
    print("ℹ️  CPU mode - install CUDA for GPU acceleration")
```

### Deliverables:
- [ ] GPU detection added to all notebooks
- [ ] XGBoost GPU params added to 03 notebook
- [ ] GPU inference enabled in 04 notebook
- [ ] Tested on Colab with GPU runtime
- [ ] Performance comparison documented (CPU vs GPU)

---

## 🔒 Phase 3: Sklearn Pipeline (Week 2)

### Priority: HIGH
**Goal:** Production-ready pipeline with preprocessing

### 3.1 Create Pipeline Module
**Create:** `src/models/pipeline.py`

```python
"""Production ML pipeline - 2025 standards."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import joblib
from pathlib import Path

class ExoplanetPipeline:
    def __init__(self, xgb_params=None, random_state=42):
        default_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'random_state': random_state,
            'tree_method': 'hist',
            'device': 'cpu'
        }
        if xgb_params:
            default_params.update(xgb_params)

        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
            ('scaler', RobustScaler()),  # Better for outliers
            ('classifier', XGBClassifier(**default_params))
        ])

    def fit(self, X, y, **fit_params):
        return self.pipeline.fit(X, y, **fit_params)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        print(f"✅ Pipeline saved: {path}")

    @classmethod
    def load(cls, path: Path):
        return joblib.load(path)
```

### 3.2 Update 03_injection_train.ipynb
**Replace training code:**
```python
from models.pipeline import ExoplanetPipeline

# Create pipeline
pipeline = ExoplanetPipeline(
    xgb_params={'device': 'cuda' if GPU_AVAILABLE else 'cpu'},
    random_state=42
)

# Train
pipeline.fit(
    X_train, y_train,
    classifier__eval_set=[(X_test, y_test)],
    classifier__verbose=False
)

# Save
pipeline.save(Path('../models/exoplanet_pipeline.pkl'))

# Evaluate
from sklearn.metrics import classification_report
y_pred = pipeline.predict_proba(X_test)[:, 1] >= 0.5
print(classification_report(y_test, y_pred))
```

### 3.3 Update 04_newdata_inference.ipynb
```python
from models.pipeline import ExoplanetPipeline

# Load pipeline (automatic preprocessing)
pipeline = ExoplanetPipeline.load('../models/exoplanet_pipeline.pkl')

# Predict (preprocessing automatic)
predictions = pipeline.predict_proba(X_new)[:, 1]
```

### Deliverables:
- [ ] `src/models/pipeline.py` created with full tests
- [ ] 03 notebook refactored to use pipeline
- [ ] 04 notebook uses loaded pipeline
- [ ] Pipeline saved and version controlled
- [ ] Tests: `pytest tests/test_pipeline.py -v`

---

## 🎯 Phase 4: Cross-Validation (Week 2-3)

### Priority: HIGH
**Goal:** Prevent data leakage with grouped CV

**Reference:** https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

### 4.1 Implement Grouped CV
**Add to `src/models/pipeline.py`:**

```python
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

def cross_validate_grouped(self, X, y, groups, n_splits=5):
    """Grouped cross-validation to prevent leakage."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                              random_state=self.pipeline.named_steps['classifier'].random_state)

    results = cross_validate(
        self.pipeline, X, y, groups=groups, cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision'],
        return_train_score=True,
        n_jobs=-1
    )

    # Print summary
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']:
        scores = results[f'test_{metric}']
        print(f"{metric:15s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return results
```

### 4.2 Update 03_injection_train.ipynb
**Add after data loading:**
```python
# Extract groups (target_id to prevent leakage)
# Ensure target_id column exists in dataset
groups = df['target_id'].values

# Grouped cross-validation
logger.info("Running stratified group k-fold cross-validation...")
cv_results = pipeline.cross_validate_grouped(X, y, groups, n_splits=5)

# Visualize results
import matplotlib.pyplot as plt
metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'average_precision']
test_scores = [cv_results[f'test_{m}'] for m in metrics]

fig, ax = plt.subplots(figsize=(12, 6))
bp = ax.boxplot(test_scores, labels=[m.replace('_', '-').upper() for m in metrics])
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Cross-Validation Performance (Stratified Group K-Fold)', fontsize=14)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('../results/cv_performance.png', dpi=300)
plt.show()
```

### Deliverables:
- [ ] GroupedCV implemented in pipeline.py
- [ ] 03 notebook uses grouped cross-validation
- [ ] Verification that same target never in train AND test
- [ ] CV performance visualization
- [ ] Documentation of leakage prevention

---

## 🌊 Phase 5: Advanced Detrending (Week 3)

### Priority: MEDIUM
**Goal:** Systematic detrending comparison

**Reference:** https://wotan.readthedocs.io/en/latest/

### 5.1 Create Detrending Module
**Create:** `src/preprocessing/detrending.py`

```python
"""Advanced detrending with wotan."""
from wotan import flatten
import numpy as np

class WotanDetrending:
    METHODS = ['biweight', 'rspline', 'hspline']

    @staticmethod
    def compare_methods(time, flux, methods=None, window_length=0.5):
        if methods is None:
            methods = WotanDetrending.METHODS

        results = {}
        for method in methods:
            flattened, trend = flatten(
                time, flux,
                method=method,
                window_length=window_length,
                return_trend=True
            )
            results[method] = {
                'flattened': flattened,
                'trend': trend,
                'rms': np.std(flattened)
            }
        return results

    @staticmethod
    def parameter_sweep(time, flux, method='biweight',
                       window_lengths=[0.2, 0.5, 1.0, 2.0, 5.0]):
        results = {}
        for wl in window_lengths:
            flattened, _ = flatten(time, flux, method=method, window_length=wl)
            results[wl] = {'rms': np.std(flattened), 'flattened': flattened}

        best_wl = min(results.keys(), key=lambda k: results[k]['rms'])
        print(f"✅ Optimal window: {best_wl} days (RMS: {results[best_wl]['rms']:.6f})")
        return results, best_wl
```

### 5.2 Update 02_bls_baseline.ipynb
**Add after data loading:**
```python
!pip install -q wotan

from preprocessing.detrending import WotanDetrending

# Compare methods
detrend_results = WotanDetrending.compare_methods(time, flux)

# Visualize
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
axes[0].plot(time, flux, 'k.', ms=1, alpha=0.5)
axes[0].set_ylabel('Original')

for i, (method, result) in enumerate(detrend_results.items(), 1):
    axes[i].plot(time, result['flattened'], '.', ms=1)
    axes[i].set_ylabel(f"{method}\\nRMS: {result['rms']:.6f}")

plt.tight_layout()
plt.savefig('../results/detrending_comparison.png', dpi=300)

# Parameter sweep
sweep_results, best_wl = WotanDetrending.parameter_sweep(time, flux)
logger.info(f"Best detrending window: {best_wl} days")
```

### Deliverables:
- [ ] Install wotan: `pip install wotan`
- [ ] Create detrending.py module
- [ ] Update 02 notebook with comparisons
- [ ] Document impact on shallow transits
- [ ] Save comparison plots to results/

---

## 📊 Phase 6: Advanced Metrics (Week 3-4)

### Priority: MEDIUM
**Goal:** Comprehensive evaluation dashboard

### 6.1 Create Metrics Module
**Create:** `src/evaluation/advanced_metrics.py`

```python
"""Advanced metrics - 2025 standards."""
from sklearn.metrics import (
    average_precision_score, brier_score_loss,
    precision_recall_curve, calibration_curve
)
import matplotlib.pyplot as plt

class AdvancedMetrics:
    @staticmethod
    def calculate_all(y_true, y_proba):
        return {
            'pr_auc': average_precision_score(y_true, y_proba),
            'brier_score': brier_score_loss(y_true, y_proba)
        }

    @staticmethod
    def plot_pr_curve(y_true, y_proba, save_path=None):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', lw=2, label=f'PR-AUC = {pr_auc:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_calibration(y_true, y_proba, save_path=None):
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        brier = brier_score_loss(y_true, y_proba)

        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect')
        plt.plot(prob_pred, prob_true, 's-', lw=2, label=f'Model (Brier={brier:.3f})')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def plot_threshold_sensitivity(y_true, y_proba, save_path=None):
        thresholds = np.linspace(0, 1, 100)
        precisions, recalls, f1_scores = [], [], []

        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))

            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

            precisions.append(p)
            recalls.append(r)
            f1_scores.append(f1)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label='Precision', lw=2)
        plt.plot(thresholds, recalls, label='Recall', lw=2)
        plt.plot(thresholds, f1_scores, label='F1-Score', lw=2)
        plt.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Default')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Threshold Sensitivity Analysis')
        plt.legend()
        plt.grid(alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
```

### 6.2 Update 05_metrics_dashboard.ipynb
```python
from evaluation.advanced_metrics import AdvancedMetrics

# Load test predictions
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Calculate metrics
metrics = AdvancedMetrics.calculate_all(y_test, y_proba)
print(f"PR-AUC: {metrics['pr_auc']:.4f}")
print(f"Brier Score: {metrics['brier_score']:.4f}")

# Generate plots
AdvancedMetrics.plot_pr_curve(y_test, y_proba, '../results/pr_curve.png')
AdvancedMetrics.plot_calibration(y_test, y_proba, '../results/calibration.png')
AdvancedMetrics.plot_threshold_sensitivity(y_test, y_proba, '../results/threshold.png')
```

### Deliverables:
- [ ] Create advanced_metrics.py module
- [ ] Update 05 notebook with new metrics
- [ ] Generate all plots in results/
- [ ] Document metric interpretation
- [ ] Compare with baseline ROC-AUC

---

## 🔍 Phase 7: SHAP Explainability (Week 4)

### Priority: MEDIUM
**Goal:** Understand feature importance

### 7.1 Add SHAP Analysis
**Install:** `pip install shap`

**Create:** `src/analysis/explainability.py`

```python
"""Model explainability with SHAP."""
import shap
import matplotlib.pyplot as plt

class SHAPAnalyzer:
    def __init__(self, model, X_train, feature_names):
        self.explainer = shap.TreeExplainer(model)
        self.shap_values = self.explainer.shap_values(X_train)
        self.feature_names = feature_names

    def plot_feature_importance(self, max_display=10, save_path=None):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            self.shap_values, features=None,
            feature_names=self.feature_names,
            max_display=max_display, show=False
        )
        plt.title("Top Features (SHAP)", fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
```

### 7.2 Update 03_injection_train.ipynb
**Add after training:**
```python
!pip install -q shap

from analysis.explainability import SHAPAnalyzer

# Extract XGBoost model
xgb_model = pipeline.pipeline.named_steps['classifier']

# Analyze
analyzer = SHAPAnalyzer(xgb_model, X_train_transformed, feature_names)
analyzer.plot_feature_importance(max_display=10, save_path='../results/shap_importance.png')
```

### Deliverables:
- [ ] Install shap
- [ ] Create explainability.py
- [ ] Add SHAP analysis to 03 notebook
- [ ] Document top 10 features
- [ ] Compare with XGBoost feature_importances_

---

## 📦 Phase 8: Probability Calibration (Week 4)

### Priority: LOW
**Goal:** Calibrated probabilities

**Reference:** https://scikit-learn.org/stable/modules/calibration.html

### 8.1 Add Calibration
**Add to pipeline.py:**
```python
from sklearn.calibration import CalibratedClassifierCV

def calibrate(self, X_cal, y_cal, method='isotonic'):
    calibrated = CalibratedClassifierCV(self.pipeline, method=method, cv=5)
    calibrated.fit(X_cal, y_cal)
    self.pipeline = calibrated
    print(f"✅ Calibrated using {method}")
```

### 8.2 Update 03_injection_train.ipynb
```python
# Split calibration set
X_train_fit, X_cal, y_train_fit, y_cal = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Train and calibrate
pipeline.fit(X_train_fit, y_train_fit)
pipeline.calibrate(X_cal, y_cal, method='isotonic')
```

### Deliverables:
- [ ] Add calibration to pipeline
- [ ] Update 03 notebook
- [ ] Compare before/after calibration curves
- [ ] Document Brier score improvement

---

## 🧪 Phase 9: End-to-End Testing (Week 5)

### Priority: HIGH
**Goal:** Verify all notebooks run in Colab

### 9.1 Create Test Suite
**Create:** `tests/test_notebooks_e2e.py`

```python
"""End-to-end notebook execution tests."""
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

@pytest.mark.parametrize("notebook", [
    "01_tap_download.ipynb",
    "02_bls_baseline.ipynb",
    "03_injection_train.ipynb",
    "04_newdata_inference.ipynb",
    "05_metrics_dashboard.ipynb"
])
def test_notebook_executes(notebook):
    with open(f'notebooks/{notebook}') as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})
```

### 9.2 Manual Colab Testing
**Checklist:**
- [ ] Upload all notebooks to Colab
- [ ] Enable GPU runtime (Runtime → Change runtime type → GPU)
- [ ] Run 01 notebook (data download)
- [ ] Run 02 notebook (BLS baseline)
- [ ] Run 03 notebook (training with GPU)
- [ ] Run 04 notebook (inference)
- [ ] Run 05 notebook (dashboard)
- [ ] Verify all results saved to results/

### Deliverables:
- [ ] E2E test suite created
- [ ] All notebooks tested in Colab
- [ ] Performance benchmarks documented
- [ ] GPU vs CPU timing comparison
- [ ] Results pushed to GitHub

---

## 📊 Success Metrics

### Code Quality:
- ✅ 100% test pass rate (currently: 14/14 ✅)
- ✅ All syntax errors fixed ✅
- [ ] 90%+ code coverage
- [ ] Zero linting errors

### Performance:
- [ ] GPU training 2-5x faster than CPU
- [ ] Pipeline reduces inference time
- [ ] Memory usage optimized

### ML Metrics:
- [ ] PR-AUC > 0.85 (better for imbalanced data)
- [ ] Brier score < 0.15 (well-calibrated)
- [ ] Cross-validated performance consistent

### Documentation:
- [x] TDD completion report ✅
- [ ] Implementation roadmap (this document)
- [ ] User guide for Colab execution
- [ ] API documentation for modules

---

## 📁 File Structure (Target)

```
exoplanet-starter/
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging_config.py       ← Phase 1
│   │   └── reproducibility.py      ← Phase 1
│   ├── models/
│   │   ├── __init__.py
│   │   └── pipeline.py             ← Phase 3
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── detrending.py           ← Phase 5
│   ├── analysis/
│   │   ├── __init__.py
│   │   └── explainability.py       ← Phase 7
│   └── evaluation/
│       ├── __init__.py
│       └── advanced_metrics.py     ← Phase 6
├── notebooks/
│   ├── 01_tap_download.ipynb       ✅ Fixed
│   ├── 02_bls_baseline.ipynb       ✅ Fixed
│   ├── 03_injection_train.ipynb    ✅ Fixed
│   ├── 04_newdata_inference.ipynb  ✅ Fixed
│   ├── 05_metrics_dashboard.ipynb  ✅ Fixed
│   ├── data_loader_colab.py        ✅ UTF-8 fixed
│   ├── quick_test.py               ✅ Created
│   └── test_02_simple.py           ✅ Created
├── tests/
│   ├── test_notebook_syntax.py     ✅ Created
│   ├── test_02_notebook_data_loading.py  ✅ Created
│   ├── test_all_notebooks_syntax.py      ✅ Created
│   ├── test_reproducibility.py     ← Phase 1
│   ├── test_pipeline.py            ← Phase 3
│   ├── test_detrending.py          ← Phase 5
│   └── test_notebooks_e2e.py       ← Phase 9
├── logs/                           ← Phase 1
├── results/                        ← All phases
└── models/                         ← Phase 3
```

---

## 🚀 Getting Started

### For Developers:
```bash
# 1. Clone repo
git clone https://github.com/exoplanet-spaceapps/exoplanet-starter.git
cd exoplanet-starter

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run tests
pytest tests/ -v

# 4. Start implementing phases
# Begin with Phase 1 (reproducibility & logging)
```

### For Colab Users:
1. Open notebook in Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells sequentially
4. Results auto-saved to results/

---

## 📞 Support & Resources

### Documentation:
- XGBoost GPU: https://xgboost.readthedocs.io/en/stable/gpu/
- Wotan detrending: https://wotan.readthedocs.io/en/latest/
- Sklearn calibration: https://scikit-learn.org/stable/modules/calibration.html
- StratifiedGroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

### Best Practices (2025):
- Always use random_state for reproducibility
- Use GroupKFold to prevent data leakage
- Prefer PR-AUC over ROC-AUC for imbalanced data
- Calibrate probabilities for production use
- Document GPU vs CPU performance
- Follow TDD: test first, implement second

---

## 📝 Notes

- This roadmap follows 2025 ML engineering best practices
- All improvements are TDD-driven with tests first
- GPU optimization is optional but recommended
- Prioritize phases 1-4 for production readiness
- Phases 5-8 enhance model quality and interpretability

**Last Updated:** 2025-09-30
**Status:** Ready for implementation
**Next Review:** After Phase 4 completion