# Comprehensive Improvements Implementation Guide

**Purpose**: This guide provides detailed code snippets and instructions for implementing ALL remaining improvements across all notebooks.

**Target**: 2025 ML Best Practices for Exoplanet Detection Pipeline

---

## 📋 Status Overview

### ✅ Completed (Phase 0-2):
- UTF-8 encoding fixes
- Data loading infrastructure
- TDD test suite
- Syntax error fixes
- Reproducibility utilities (`set_random_seeds`)
- Logging utilities
- GPU detection utilities

### 🚧 In Progress (Phase 3-9):
This guide covers implementation of:
- Phase 3: Sklearn Pipeline
- Phase 4: StratifiedGroupKFold
- Phase 5: Wotan detrending
- Phase 6: Advanced metrics
- Phase 7: SHAP explainability
- Phase 8: Probability calibration
- GPU optimization for 03/04
- `random_state=42` everywhere

---

## 🎯 Implementation Priority

### HIGH PRIORITY (Must Have):
1. **03_injection_train.ipynb**: GPU + Pipeline + GroupKFold
2. **04_newdata_inference.ipynb**: GPU inference
3. **random_state=42**: All notebooks

### MEDIUM PRIORITY (Should Have):
4. **03_injection_train.ipynb**: SHAP + Calibration
5. **05_metrics_dashboard.ipynb**: PR-AUC + Brier + Calibration curves
6. **02_bls_baseline.ipynb**: Wotan detrending comparison

### LOW PRIORITY (Nice to Have):
7. Threshold sensitivity curves
8. Multi-detrending comparison plots
9. Advanced feature engineering

---

## 📦 Required Package Installations

Add to **first cell** of each notebook:

```python
# Install all required packages
!pip install -q numpy==1.26.4 pandas scikit-learn==1.5.0
!pip install -q xgboost==2.1.0  # For GPU support
!pip install -q lightkurve astroquery transitleastsquares
!pip install -q wotan  # Advanced detrending
!pip install -q shap  # Explainability
!pip install -q matplotlib seaborn

# Restart runtime after installation (Colab only)
print("✅ Packages installed - Please restart runtime if in Colab")
```

---

## 🔧 03_injection_train.ipynb Updates

### Cell 1: GPU Setup & Reproducibility

Insert **after package imports**:

```python
# ================================================================
# GPU Setup & Reproducibility (Phase 1-2)
# ================================================================
import sys
from pathlib import Path

# Add src to path
src_path = Path('../src') if 'COLAB' not in str(Path.cwd()) else Path('/content/exoplanet-starter/src')
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import utilities
try:
    from utils import (
        set_random_seeds,
        detect_gpu,
        get_xgboost_gpu_params,
        log_gpu_info,
        setup_logger,
        get_log_file_path
    )

    # Set random seeds for reproducibility
    set_random_seeds(42)
    print("✅ Random seeds set to 42")

    # Setup logging
    log_file = get_log_file_path("03_injection_train")
    logger = setup_logger("03_injection", log_file=log_file)
    logger.info("Starting 03_injection_train.ipynb")

    # Detect GPU and get XGBoost params
    gpu_info = detect_gpu()
    log_gpu_info(logger)

    # Get XGBoost GPU parameters (XGBoost 2.x API)
    xgb_gpu_params = get_xgboost_gpu_params()
    logger.info(f"XGBoost GPU params: {xgb_gpu_params}")

    print(f"\n🖥️ GPU Available: {'✅' if gpu_info['available'] else '❌'}")
    if gpu_info['available']:
        print(f"   GPU: {gpu_info['device_name']}")
        print(f"   XGBoost GPU: {xgb_gpu_params}")
    else:
        print("   Using CPU for training")

except ImportError as e:
    print(f"⚠️ Could not import utilities: {e}")
    print("   Continuing without utilities...")

    # Fallback: Set random seeds manually
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)

    xgb_gpu_params = {'tree_method': 'hist', 'device': 'cpu'}
    logger = None

print("\n✅ Setup complete!\n")
```

### Cell 2: Sklearn Pipeline Creation (Phase 3)

Insert **before model training code**:

```python
# ================================================================
# Sklearn Pipeline with Preprocessing (Phase 3)
# ================================================================
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import xgboost as xgb

def create_exoplanet_pipeline(
    numerical_features,
    xgb_params=None,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
):
    """
    Create complete exoplanet detection pipeline

    Pipeline steps:
    1. SimpleImputer: Fill missing values with median
    2. RobustScaler: Scale features (robust to outliers)
    3. XGBClassifier: Train model with GPU support

    Args:
        numerical_features: List of numerical feature names
        xgb_params: Dict of XGBoost parameters (including GPU)
        n_estimators: Number of trees
        max_depth: Max tree depth
        learning_rate: Learning rate
        random_state: Random seed

    Returns:
        sklearn.pipeline.Pipeline
    """
    if xgb_params is None:
        xgb_params = {}

    # Numerical preprocessing pipeline
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', RobustScaler())  # Robust to outliers (better than StandardScaler)
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='drop'  # Drop non-numerical features
    )

    # Complete pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            **xgb_params,  # GPU params from Phase 2
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,  # Reproducibility
            eval_metric='aucpr',  # PR-AUC (better for imbalanced data)
            early_stopping_rounds=10
        ))
    ])

    return pipeline

# Define numerical features
numerical_features = [
    'bls_period', 'bls_depth_ppm', 'bls_snr', 'bls_duration_hours',
    'tls_period', 'tls_depth_ppm', 'tls_sde', 'tls_duration_hours',
    'period_ratio', 'depth_ratio', 'snr_ratio',
    'period_diff_pct', 'depth_diff_pct', 'snr_improvement'
]

# Create pipeline with GPU support
pipeline = create_exoplanet_pipeline(
    numerical_features=numerical_features,
    xgb_params=xgb_gpu_params,  # From GPU setup cell
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

print("✅ Pipeline created successfully!")
print(f"\nPipeline structure:")
print(f"  1. Preprocessing:")
print(f"     - SimpleImputer (strategy=median)")
print(f"     - RobustScaler (robust to outliers)")
print(f"  2. Classifier:")
print(f"     - XGBClassifier")
print(f"     - GPU device: {xgb_gpu_params.get('device', 'cpu')}")
print(f"     - n_estimators: 100")
print(f"     - eval_metric: aucpr (PR-AUC)")
print(f"     - random_state: 42")
print()
```

### Cell 3: StratifiedGroupKFold Cross-Validation (Phase 4)

Insert **before final model training**:

```python
# ================================================================
# StratifiedGroupKFold Cross-Validation (Phase 4)
# Prevents data leakage by ensuring same target_id not in train+test
# ================================================================
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.metrics import make_scorer, precision_recall_curve, auc
import pandas as pd
import numpy as np

def pr_auc_score(y_true, y_pred_proba):
    """Compute Precision-Recall AUC"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)

# Define scoring metrics
scoring = {
    'pr_auc': make_scorer(pr_auc_score, needs_proba=True),
    'roc_auc': 'roc_auc',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

# Use StratifiedGroupKFold to prevent data leakage
# - Stratified: Maintains label distribution across folds
# - Grouped: Ensures same target_id not in both train and test
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

print("🔄 Starting StratifiedGroupKFold Cross-Validation...")
print(f"   Number of splits: 5")
print(f"   Grouping by: target_id")
print(f"   Stratifying by: label")
print(f"   Shuffle: True (random_state=42)")
print()

# Perform cross-validation
cv_results = cross_validate(
    pipeline,
    X=features_df[numerical_features],
    y=features_df['label'],
    groups=features_df['target_id'],  # Group by target_id to prevent leakage
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1,  # Use all CPU cores
    verbose=1
)

# Display results
print("\n" + "="*60)
print("📊 Cross-Validation Results")
print("="*60)

results_summary = []
for metric in ['pr_auc', 'roc_auc', 'precision', 'recall', 'f1']:
    train_scores = cv_results[f'train_{metric}']
    test_scores = cv_results[f'test_{metric}']

    results_summary.append({
        'Metric': metric.upper().replace('_', '-'),
        'Train Mean': f"{train_scores.mean():.4f}",
        'Train Std': f"{train_scores.std():.4f}",
        'Test Mean': f"{test_scores.mean():.4f}",
        'Test Std': f"{test_scores.std():.4f}",
        'Overfitting': f"{(train_scores.mean() - test_scores.mean()):.4f}"
    })

results_df = pd.DataFrame(results_summary)
print(results_df.to_string(index=False))

# Visualize CV results
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Boxplot of test scores
metrics_to_plot = ['pr_auc', 'roc_auc', 'precision', 'recall', 'f1']
test_scores_list = [cv_results[f'test_{m}'] for m in metrics_to_plot]

bp = axes[0].boxplot(test_scores_list, labels=[m.upper().replace('_', '-') for m in metrics_to_plot])
axes[0].set_title('Cross-Validation Test Scores Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Score', fontsize=12)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1.05])

# Train vs Test comparison
train_means = [cv_results[f'train_{m}'].mean() for m in metrics_to_plot]
test_means = [cv_results[f'test_{m}'].mean() for m in metrics_to_plot]

x = np.arange(len(metrics_to_plot))
width = 0.35

axes[1].bar(x - width/2, train_means, width, label='Train', alpha=0.8, color='skyblue')
axes[1].bar(x + width/2, test_means, width, label='Test', alpha=0.8, color='salmon')
axes[1].set_xticks(x)
axes[1].set_xticklabels([m.upper().replace('_', '-') for m in metrics_to_plot], rotation=45, ha='right')
axes[1].set_title('Train vs Test Scores Comparison', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Score', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].set_ylim([0, 1.05])

plt.tight_layout()
plt.show()

# Key findings
pr_auc_mean = cv_results['test_pr_auc'].mean()
pr_auc_std = cv_results['test_pr_auc'].std()

print(f"\n💡 Key Findings:")
print(f"   Primary Metric (PR-AUC): {pr_auc_mean:.4f} ± {pr_auc_std:.4f}")
print(f"   ROC-AUC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
print(f"   F1-Score: {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")

if (train_means[0] - test_means[0]) > 0.1:
    print(f"\n⚠️ Warning: Significant overfitting detected (train-test gap: {train_means[0] - test_means[0]:.3f})")
else:
    print(f"\n✅ No significant overfitting detected")

print("\n✅ Cross-validation complete!\n")
```

### Cell 4: Train Final Model

```python
# ================================================================
# Train Final Model on All Data
# ================================================================
print("🎯 Training final model on complete dataset...")

# Train pipeline
pipeline.fit(
    features_df[numerical_features],
    features_df['label']
)

print("✅ Final model trained!\n")
```

### Cell 5: SHAP Explainability (Phase 7)

Insert **after final model training**:

```python
# ================================================================
# SHAP Explainability Analysis (Phase 7)
# ================================================================
try:
    import shap

    print("🔍 Performing SHAP analysis...")

    # Extract the trained XGBoost model from pipeline
    xgb_model = pipeline.named_steps['classifier']

    # Get preprocessed features
    X_preprocessed = pipeline.named_steps['preprocessor'].transform(
        features_df[numerical_features]
    )

    # Create SHAP TreeExplainer (optimized for tree models)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_preprocessed)

    # SHAP Summary Plot (Bar) - Feature importance
    print("\n📊 Feature Importance (SHAP):")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_preprocessed,
        feature_names=numerical_features,
        plot_type="bar",
        show=False
    )
    ax.set_title('Feature Importance (mean |SHAP value|)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # SHAP Summary Plot (Detailed) - Impact direction
    print("\n📊 Feature Impact (SHAP):")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_preprocessed,
        feature_names=numerical_features,
        show=False
    )
    ax.set_title('SHAP Values for Each Feature', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Calculate and display feature importance ranking
    feature_importance_shap = pd.DataFrame({
        'feature': numerical_features,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    print("\n📈 Top 10 Most Important Features (SHAP):")
    print(feature_importance_shap.head(10).to_string(index=False))

    # Compare with XGBoost's built-in feature importance
    xgb_feature_importance = pd.DataFrame({
        'feature': numerical_features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n📈 Top 10 Features (XGBoost built-in):")
    print(xgb_feature_importance.head(10).to_string(index=False))

    print("\n✅ SHAP analysis complete!")
    print("\n💡 Interpretation:")
    print("   - Red points: High feature value")
    print("   - Blue points: Low feature value")
    print("   - X-axis: Positive SHAP → increases exoplanet probability")
    print("   - X-axis: Negative SHAP → decreases exoplanet probability\n")

except ImportError:
    print("⚠️ SHAP not installed, skipping explainability analysis")
    print("   Install with: !pip install shap\n")
except Exception as e:
    print(f"⚠️ SHAP analysis failed: {e}\n")
```

### Cell 6: Probability Calibration (Phase 8)

Insert **after SHAP analysis**:

```python
# ================================================================
# Probability Calibration (Phase 8)
# ================================================================
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

print("🔧 Training calibrated model...")

# Create calibrated classifier
# - method='isotonic': Better for tree models (non-parametric)
# - cv=5: 5-fold cross-validation for calibration
calibrated_pipeline = CalibratedClassifierCV(
    pipeline,
    method='isotonic',  # Isotonic regression (best for tree models)
    cv=5,  # 5-fold CV
    n_jobs=-1
)

# Train calibrated model
calibrated_pipeline.fit(
    features_df[numerical_features],
    features_df['label']
)

print("✅ Calibration training complete!\n")

# Get predictions before and after calibration
y_true = features_df['label']
y_pred_proba_before = pipeline.predict_proba(features_df[numerical_features])[:, 1]
y_pred_proba_after = calibrated_pipeline.predict_proba(features_df[numerical_features])[:, 1]

# Calculate Brier scores
brier_before = brier_score_loss(y_true, y_pred_proba_before)
brier_after = brier_score_loss(y_true, y_pred_proba_after)
brier_improvement = ((brier_before - brier_after) / brier_before) * 100

print("="*60)
print("📊 Calibration Results")
print("="*60)
print(f"Brier Score (lower is better):")
print(f"   Before calibration: {brier_before:.4f}")
print(f"   After calibration:  {brier_after:.4f}")
print(f"   Improvement: {brier_improvement:.2f}%")
print()

# Plot calibration curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Calibration curve
ax = axes[0]
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')

# Before calibration
prob_true_before, prob_pred_before = calibration_curve(
    y_true,
    y_pred_proba_before,
    n_bins=10
)
ax.plot(prob_pred_before, prob_true_before, 's-',
        linewidth=2, markersize=10,
        label=f'Before (Brier={brier_before:.4f})',
        color='salmon')

# After calibration
prob_true_after, prob_pred_after = calibration_curve(
    y_true,
    y_pred_proba_after,
    n_bins=10
)
ax.plot(prob_pred_after, prob_true_after, 'o-',
        linewidth=2, markersize=10,
        label=f'After (Brier={brier_after:.4f})',
        color='skyblue')

ax.set_xlabel('Mean Predicted Probability', fontsize=12)
ax.set_ylabel('Fraction of Positives', fontsize=12)
ax.set_title('Calibration Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.05, 1.05])
ax.set_ylim([-0.05, 1.05])

# Histogram of predicted probabilities
ax = axes[1]
ax.hist(y_pred_proba_before, bins=20, alpha=0.6, label='Before', color='salmon', edgecolor='black')
ax.hist(y_pred_proba_after, bins=20, alpha=0.6, label='After', color='skyblue', edgecolor='black')
ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Predicted Probability Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("\n💡 Calibration Interpretation:")
print("   - Calibrated probabilities are more reliable")
print("   - Useful for decision-making (e.g., follow-up observations)")
print("   - Lower Brier score indicates better calibration")
print("\n✅ Probability calibration complete!\n")
```

### Cell 7: Save Model

```python
# ================================================================
# Save Trained Models
# ================================================================
import joblib
from pathlib import Path

# Create model directory
model_dir = Path('../models')
model_dir.mkdir(exist_ok=True)

# Save uncalibrated pipeline
pipeline_path = model_dir / 'exoplanet_pipeline_v1.joblib'
joblib.dump(pipeline, pipeline_path)
print(f"✅ Saved pipeline: {pipeline_path}")

# Save calibrated pipeline
calibrated_path = model_dir / 'exoplanet_pipeline_calibrated_v1.joblib'
joblib.dump(calibrated_pipeline, calibrated_path)
print(f"✅ Saved calibrated pipeline: {calibrated_path}")

# Save feature names
feature_names_path = model_dir / 'feature_names.txt'
with open(feature_names_path, 'w') as f:
    f.write('\n'.join(numerical_features))
print(f"✅ Saved feature names: {feature_names_path}")

# Save model metadata
import json
metadata = {
    'model_type': 'XGBClassifier',
    'features': numerical_features,
    'n_features': len(numerical_features),
    'preprocessing': 'SimpleImputer + RobustScaler',
    'calibration': 'Isotonic (5-fold CV)',
    'xgb_params': xgb_gpu_params,
    'cv_pr_auc_mean': float(cv_results['test_pr_auc'].mean()),
    'cv_pr_auc_std': float(cv_results['test_pr_auc'].std()),
    'brier_score_before': float(brier_before),
    'brier_score_after': float(brier_after),
    'random_state': 42
}

metadata_path = model_dir / 'model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Saved metadata: {metadata_path}")

print("\n✅ All models saved successfully!\n")
```

---

## 📝 04_newdata_inference.ipynb Updates

### Add GPU Support for Inference

Insert at the beginning (after imports):

```python
# GPU Setup for Inference
import sys
from pathlib import Path

src_path = Path('../src') if 'COLAB' not in str(Path.cwd()) else Path('/content/exoplanet-starter/src')
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from utils import detect_gpu, get_pytorch_device, set_random_seeds

    set_random_seeds(42)
    gpu_info = detect_gpu()
    device = get_pytorch_device()

    print(f"🖥️ GPU Available: {'✅' if gpu_info['available'] else '❌'}")
    if gpu_info['available']:
        print(f"   Device: {device}")
        print(f"   GPU: {gpu_info['device_name']}")
except ImportError:
    device = 'cpu'
    print("⚠️ Using CPU for inference")

print("\n✅ Inference setup complete!\n")
```

---

## 📊 05_metrics_dashboard.ipynb Updates

### Add Advanced Metrics (Phase 6)

```python
# ================================================================
# Advanced Metrics Dashboard (Phase 6)
# ================================================================
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_curve,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import numpy as np

def plot_advanced_metrics(y_true, y_pred_proba, y_pred_labels=None):
    """
    Plot comprehensive evaluation metrics

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        y_pred_labels: Predicted labels (optional, will threshold at 0.5)
    """
    if y_pred_labels is None:
        y_pred_labels = (y_pred_proba >= 0.5).astype(int)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. PR Curve
    ax1 = fig.add_subplot(gs[0, 0])
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    ax1.plot(recall, precision, linewidth=2, label=f'PR-AUC = {pr_auc:.4f}')
    ax1.fill_between(recall, precision, alpha=0.2)
    ax1.set_xlabel('Recall', fontsize=11)
    ax1.set_ylabel('Precision', fontsize=11)
    ax1.set_title('Precision-Recall Curve', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1.05])

    # 2. ROC Curve
    ax2 = fig.add_subplot(gs[0, 1])
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    ax2.plot(fpr, tpr, linewidth=2, label=f'ROC-AUC = {roc_auc:.4f}')
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax2.fill_between(fpr, tpr, alpha=0.2)
    ax2.set_xlabel('False Positive Rate', fontsize=11)
    ax2.set_ylabel('True Positive Rate', fontsize=11)
    ax2.set_title('ROC Curve', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1.05])

    # 3. Calibration Curve
    ax3 = fig.add_subplot(gs[0, 2])
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    brier = brier_score_loss(y_true, y_pred_proba)
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect')
    ax3.plot(prob_pred, prob_true, 's-', linewidth=2, markersize=8,
             label=f'Model (Brier={brier:.4f})')
    ax3.set_xlabel('Mean Predicted Probability', fontsize=11)
    ax3.set_ylabel('Fraction of Positives', fontsize=11)
    ax3.set_title('Calibration Curve', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1.05])

    # 4. Threshold Sensitivity (Precision)
    ax4 = fig.add_subplot(gs[1, 0])
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        if y_pred_thresh.sum() > 0:  # Avoid division by zero
            prec = precision_score(y_true, y_pred_thresh, zero_division=0)
        else:
            prec = 0
        precisions.append(prec)
    ax4.plot(thresholds, precisions, linewidth=2)
    ax4.axvline(0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax4.set_xlabel('Threshold', fontsize=11)
    ax4.set_ylabel('Precision', fontsize=11)
    ax4.set_title('Precision vs Threshold', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1.05])

    # 5. Threshold Sensitivity (Recall)
    ax5 = fig.add_subplot(gs[1, 1])
    recalls = []
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        rec = recall_score(y_true, y_pred_thresh, zero_division=0)
        recalls.append(rec)
    ax5.plot(thresholds, recalls, linewidth=2, color='orange')
    ax5.axvline(0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax5.set_xlabel('Threshold', fontsize=11)
    ax5.set_ylabel('Recall', fontsize=11)
    ax5.set_title('Recall vs Threshold', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1.05])

    # 6. F1-Score vs Threshold
    ax6 = fig.add_subplot(gs[1, 2])
    from sklearn.metrics import f1_score
    f1_scores = []
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
        f1_scores.append(f1)
    optimal_thresh = thresholds[np.argmax(f1_scores)]
    ax6.plot(thresholds, f1_scores, linewidth=2, color='green')
    ax6.axvline(optimal_thresh, color='b', linestyle='--', alpha=0.5,
                label=f'Optimal ({optimal_thresh:.2f})')
    ax6.axvline(0.5, color='r', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax6.set_xlabel('Threshold', fontsize=11)
    ax6.set_ylabel('F1-Score', fontsize=11)
    ax6.set_title('F1-Score vs Threshold', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.set_ylim([0, 1.05])

    # 7. Confusion Matrix
    ax7 = fig.add_subplot(gs[2, 0])
    cm = confusion_matrix(y_true, y_pred_labels)
    im = ax7.imshow(cm, cmap='Blues', aspect='auto')
    ax7.set_xticks([0, 1])
    ax7.set_yticks([0, 1])
    ax7.set_xticklabels(['Negative', 'Positive'])
    ax7.set_yticklabels(['Negative', 'Positive'])
    ax7.set_xlabel('Predicted', fontsize=11)
    ax7.set_ylabel('True', fontsize=11)
    ax7.set_title('Confusion Matrix', fontsize=13, fontweight='bold')

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax7.text(j, i, cm[i, j],
                           ha="center", va="center", color="black", fontsize=14)

    plt.colorbar(im, ax=ax7)

    # 8. Predicted Probability Distribution
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.6, label='Negative', color='salmon', edgecolor='black')
    ax8.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.6, label='Positive', color='skyblue', edgecolor='black')
    ax8.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.5)')
    ax8.set_xlabel('Predicted Probability', fontsize=11)
    ax8.set_ylabel('Count', fontsize=11)
    ax8.set_title('Predicted Probability Distribution', fontsize=13, fontweight='bold')
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3, axis='y')

    # 9. Classification Report (Text)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    report = classification_report(y_true, y_pred_labels, output_dict=True)

    report_text = f"""
    Classification Report

    Class 0 (Negative):
      Precision: {report['0']['precision']:.4f}
      Recall:    {report['0']['recall']:.4f}
      F1-Score:  {report['0']['f1-score']:.4f}

    Class 1 (Positive):
      Precision: {report['1']['precision']:.4f}
      Recall:    {report['1']['recall']:.4f}
      F1-Score:  {report['1']['f1-score']:.4f}

    Overall:
      Accuracy:  {report['accuracy']:.4f}
      Macro Avg: {report['macro avg']['f1-score']:.4f}

    Optimal Threshold: {optimal_thresh:.4f}
    """

    ax9.text(0.1, 0.5, report_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
             facecolor='wheat', alpha=0.3))

    plt.suptitle('Comprehensive Model Evaluation Dashboard', fontsize=16, fontweight='bold', y=0.995)
    plt.show()

    # Print summary
    print("\n" + "="*60)
    print("📊 Model Performance Summary")
    print("="*60)
    print(f"PR-AUC:       {pr_auc:.4f}")
    print(f"ROC-AUC:      {roc_auc:.4f}")
    print(f"Brier Score:  {brier:.4f}")
    print(f"Precision:    {report['1']['precision']:.4f}")
    print(f"Recall:       {report['1']['recall']:.4f}")
    print(f"F1-Score:     {report['1']['f1-score']:.4f}")
    print(f"Accuracy:     {report['accuracy']:.4f}")
    print(f"\nOptimal Threshold: {optimal_thresh:.4f}")
    print("="*60 + "\n")

# Example usage:
# plot_advanced_metrics(y_true, y_pred_proba)
```

---

## 🌟 02_bls_baseline.ipynb Updates

### Add Wotan Detrending Comparison (Phase 5)

Insert before BLS/TLS search:

```python
# ================================================================
# Wotan Detrending Methods Comparison (Phase 5)
# ================================================================
try:
    from wotan import flatten

    def compare_detrending_methods(lc, methods=['biweight', 'rspline', 'hspline'], window_length=0.5):
        """
        Compare different detrending methods from wotan

        Args:
            lc: lightkurve LightCurve object
            methods: List of wotan methods to compare
            window_length: Window length in days

        Returns:
            dict: Dictionary of detrended lightcurves
        """
        time = lc.time.value
        flux = lc.flux.value

        results = {}

        for method in methods:
            try:
                # Apply wotan detrending
                flux_detrended, trend = flatten(
                    time,
                    flux,
                    method=method,
                    window_length=window_length,
                    return_trend=True
                )

                # Create new LightCurve object
                lc_detrended = lk.LightCurve(
                    time=time,
                    flux=flux_detrended
                )

                results[method] = {
                    'lightcurve': lc_detrended,
                    'trend': trend,
                    'std': np.std(flux_detrended),
                    'mad': np.median(np.abs(flux_detrended - np.median(flux_detrended)))
                }

                print(f"✅ {method:12s}: std={results[method]['std']:.6f}, MAD={results[method]['mad']:.6f}")

            except Exception as e:
                print(f"⚠️ {method} failed: {e}")

        return results

    print("🔧 Comparing detrending methods...")
    print(f"   Methods: biweight, rspline, hspline")
    print(f"   Window length: 0.5 days\n")

    detrending_results = compare_detrending_methods(
        lc_clean,
        methods=['biweight', 'rspline', 'hspline'],
        window_length=0.5
    )

    # Visualize comparison
    fig, axes = plt.subplots(len(detrending_results) + 1, 1, figsize=(14, 4*(len(detrending_results)+1)))

    # Original
    axes[0].plot(lc_clean.time.value, lc_clean.flux.value, '.', markersize=1, alpha=0.5)
    axes[0].set_title('Original Light Curve', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Flux')
    axes[0].grid(True, alpha=0.3)

    # Detrended versions
    for idx, (method, result) in enumerate(detrending_results.items(), 1):
        lc_det = result['lightcurve']
        axes[idx].plot(lc_det.time.value, lc_det.flux.value, '.', markersize=1, alpha=0.5)
        axes[idx].set_title(f'{method.capitalize()} Detrending (std={result["std"]:.6f})',
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Normalized Flux')
        axes[idx].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (BTJD)')

    plt.tight_layout()
    plt.show()

    print("\n✅ Detrending comparison complete!")
    print("\n💡 Recommendation: Use 'biweight' for robust detrending")
    print("   - Biweight: Most robust to outliers")
    print("   - Rspline: Smooth polynomial fit")
    print("   - Hspline: Huber-weighted spline\n")

except ImportError:
    print("⚠️ wotan not installed, skipping advanced detrending")
    print("   Install with: !pip install wotan\n")
```

---

## ✅ Implementation Checklist

### For each notebook:

- [ ] **01_tap_download.ipynb**: Already complete ✅
- [ ] **02_bls_baseline.ipynb**:
  - [ ] Add reproducibility setup (Cell 4 already done ✅)
  - [ ] Add wotan detrending comparison
  - [ ] Update logging

- [ ] **03_injection_train.ipynb**:
  - [ ] Add GPU setup + reproducibility
  - [ ] Create sklearn Pipeline
  - [ ] Implement StratifiedGroupKFold
  - [ ] Add SHAP explainability
  - [ ] Add probability calibration
  - [ ] Save models with metadata

- [ ] **04_newdata_inference.ipynb**:
  - [ ] Add GPU support for inference
  - [ ] Load calibrated model
  - [ ] Add reproducibility setup

- [ ] **05_metrics_dashboard.ipynb**:
  - [ ] Add advanced metrics plots
  - [ ] Add calibration curves
  - [ ] Add threshold sensitivity analysis

### Global Tasks:
- [ ] Verify `random_state=42` in ALL random operations
- [ ] Test all notebooks in sequence (local → Colab)
- [ ] Verify GPU acceleration works on Colab
- [ ] Update README with new features
- [ ] Commit and push all changes

---

## 📚 Additional Resources

- XGBoost 2.x GPU: https://xgboost.readthedocs.io/en/release_2.0.0/gpu/
- Wotan documentation: https://wotan.readthedocs.io/en/latest/
- SHAP documentation: https://shap.readthedocs.io/
- Scikit-learn Calibration: https://scikit-learn.org/stable/modules/calibration.html
- StratifiedGroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html

---

*Last Updated: 2025-09-30*
*Status: Phase 3-9 Implementation Guide*