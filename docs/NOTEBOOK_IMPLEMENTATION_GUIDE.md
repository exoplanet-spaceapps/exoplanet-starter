# Notebook Implementation Guide - Missing Features

## 📋 Overview

This guide provides ready-to-use code snippets for implementing the missing features in notebooks 03, 04, and 05 as identified by the user.

All implementations follow TDD principles and are tested in:
- `tests/test_model_card.py`
- `tests/test_calibration_curves.py`
- `tests/test_output_schema.py`
- `tests/test_latency_metrics.py`

---

## 🎯 Notebook 03: `03_injection_train.ipynb`

### Missing Features:
1. ✅ Platt calibration option (alongside Isotonic)
2. ✅ Save calibration curves to `/reports/`
3. ✅ Generate and save Model Card to `/reports/`

### Implementation:

#### Add After Existing Calibration Code:

```python
# Cell: Probability Calibration (Enhanced with Platt + Model Card)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from utils import plot_calibration_curves, create_model_card, save_model_card
from pathlib import Path

print("🔧 Training Calibrated Models...")

# Train both Isotonic and Platt calibration
calibrated_isotonic = CalibratedClassifierCV(
    pipeline,
    method='isotonic',  # Best for tree models
    cv=5
)
calibrated_isotonic.fit(X_train, y_train)

calibrated_platt = CalibratedClassifierCV(
    pipeline,
    method='sigmoid',  # Platt scaling
    cv=5
)
calibrated_platt.fit(X_train, y_train)

# Get predictions
y_pred_uncalib = pipeline.predict_proba(X_test)[:, 1]
y_pred_isotonic = calibrated_isotonic.predict_proba(X_test)[:, 1]
y_pred_platt = calibrated_platt.predict_proba(X_test)[:, 1]

# Calculate Brier scores
brier_uncalib = brier_score_loss(y_test, y_pred_uncalib)
brier_isotonic = brier_score_loss(y_test, y_pred_isotonic)
brier_platt = brier_score_loss(y_test, y_pred_platt)

print(f"\n📊 Calibration Results:")
print(f"   Uncalibrated Brier Score: {brier_uncalib:.4f}")
print(f"   Isotonic Brier Score:     {brier_isotonic:.4f} ({((brier_uncalib-brier_isotonic)/brier_uncalib*100):.1f}% improvement)")
print(f"   Platt Brier Score:        {brier_platt:.4f} ({((brier_uncalib-brier_platt)/brier_uncalib*100):.1f}% improvement)")

# Save calibration curves
plot_calibration_curves(
    y_true=y_test,
    predictions={
        "Uncalibrated": y_pred_uncalib,
        "Isotonic": y_pred_isotonic,
        "Platt (Sigmoid)": y_pred_platt
    },
    output_path=Path("reports/calibration_curves.png")
)

# Determine best calibration method
best_method = "isotonic" if brier_isotonic <= brier_platt else "platt"
best_score = min(brier_isotonic, brier_platt)
print(f"\n✅ Best calibration method: {best_method.upper()} (Brier: {best_score:.4f})")
```

#### Add Model Card Generation:

```python
# Cell: Generate Model Card

from datetime import datetime

# Collect metrics
from sklearn.metrics import roc_auc_score, average_precision_score

metrics = {
    "roc_auc": roc_auc_score(y_test, y_pred_isotonic),
    "pr_auc": average_precision_score(y_test, y_pred_isotonic),
    "brier_score_uncalibrated": brier_uncalib,
    "brier_score_calibrated": brier_isotonic,
    "brier_improvement_pct": ((brier_uncalib - brier_isotonic) / brier_uncalib * 100),
    "test_accuracy": (y_test == (y_pred_isotonic > 0.5)).mean()
}

# Create model card
model_card = create_model_card(
    model_name="XGBoost_Exoplanet_Detector",
    model_version="v1.0.0",
    training_date=datetime.now().strftime("%Y-%m-%d"),
    metrics=metrics,
    features=numerical_features,  # From your feature engineering
    calibration_method="isotonic",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    },
    dataset_info={
        "name": "NASA TOI + KOI Combined",
        "n_samples": len(X_train) + len(X_test),
        "n_positives": (y_train.sum() + y_test.sum()),
        "n_negatives": (len(y_train) - y_train.sum() + len(y_test) - y_test.sum()),
        "test_size": "20%"
    },
    additional_notes="Trained with StratifiedGroupKFold to prevent data leakage"
)

# Save model card
save_model_card(model_card, Path("reports/model_card.json"))

print("✅ Model Card saved to: reports/model_card.json")
print("✅ Calibration curves saved to: reports/calibration_curves.png")
```

---

## 📁 Notebook 04: `04_newdata_inference.ipynb`

### Missing Features:
1. ✅ CSV export with standardized schema
2. ✅ Provenance tracking (versions, timestamps, parameters)

### Implementation:

#### Replace Existing Inference Code:

```python
# Cell: Inference with Standardized Output & Provenance

from utils import (
    create_candidate_dataframe,
    export_candidates_csv,
    export_candidates_jsonl,
    create_provenance_record,
    save_provenance
)
from datetime import datetime
from pathlib import Path

print("🔮 Running Inference on New Data...")

# Load model (assumed already loaded as 'calibrated_model')
y_pred_proba = calibrated_model.predict_proba(features_df[numerical_features])[:, 1]
y_pred_uncalib = pipeline.predict_proba(features_df[numerical_features])[:, 1]

# Create BLS results dict
bls_results = {}
for idx, row in features_df.iterrows():
    target_id = row['target_id']
    bls_results[target_id] = {
        "period": row.get('bls_period', np.nan),
        "duration": row.get('bls_duration', np.nan),
        "depth": row.get('bls_depth_ppm', np.nan),
        "snr": row.get('snr', np.nan),
        "power": row.get('power', np.nan),
        "sector": row.get('sector', ""),
        "is_eb": row.get('is_eb', False),
        "toi_match": row.get('toi_crossmatch', ""),
        "flags": row.get('quality_flags', ""),
        "url": row.get('data_source_url', "")
    }

# Model predictions dict
model_predictions = {
    **{target_id: score for target_id, score in zip(features_df['target_id'], y_pred_proba)},
    **{f"{target_id}_uncalib": score for target_id, score in zip(features_df['target_id'], y_pred_uncalib)}
}

# Create standardized DataFrame
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
candidates_df = create_candidate_dataframe(
    target_ids=features_df['target_id'].tolist(),
    missions=features_df['mission'].tolist(),
    bls_results=bls_results,
    model_predictions=model_predictions,
    additional_features=features_df if 'pscomp_pl_rade' in features_df.columns else None,
    run_id=run_id,
    model_version="v1.0.0"
)

# Export to CSV
timestamp = datetime.now().strftime("%Y%m%d")
csv_path = export_candidates_csv(candidates_df, timestamp=timestamp)

# Also export JSONL
jsonl_path = export_candidates_jsonl(candidates_df, timestamp=timestamp)

print(f"\n✅ Exported {len(candidates_df)} candidates")
print(f"   CSV: {csv_path}")
print(f"   JSONL: {jsonl_path}")
```

#### Add Provenance Tracking:

```python
# Cell: Create Provenance Record

import importlib.metadata

# Create comprehensive provenance
provenance = create_provenance_record(
    run_id=run_id,
    data_source="MAST TAP" if IN_COLAB else "Local CSV",
    mission="TESS",  # Or dynamically from data
    query_params={
        "query_type": "cone_search" if 'ra' in features_df.columns else "target_id_list",
        "n_targets": len(features_df),
        "sectors": features_df['sector'].unique().tolist() if 'sector' in features_df.columns else []
    },
    model_info={
        "version": "v1.0.0",
        "path": "models/xgb_calibrated_model.pkl",
        "calibration": "isotonic",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1
        },
        "random_state": 42
    },
    processing_steps=[
        "1. Light curve download",
        "2. BLS feature extraction",
        "3. Preprocessing (impute + scale)",
        "4. XGBoost prediction",
        "5. Isotonic calibration",
        "6. Export to CSV/JSONL"
    ],
    output_files=[str(csv_path), str(jsonl_path)],
    additional_metadata={
        "notebook": "04_newdata_inference.ipynb",
        "execution_environment": "Google Colab" if IN_COLAB else "Local",
        "n_candidates": len(candidates_df),
        "high_confidence_candidates": (candidates_df['model_score'] > 0.8).sum()
    }
)

# Save provenance
save_provenance(provenance, Path("outputs/provenance.yaml"))

print("\n✅ Provenance tracking saved to: outputs/provenance.yaml")
print(f"\n📦 Package Versions:")
for pkg in ['lightkurve', 'numpy', 'xgboost', 'scikit-learn']:
    print(f"   {pkg}: {provenance['dependencies'][pkg]}")
```

---

## 📊 Notebook 05: `05_metrics_dashboard.ipynb`

### Missing Features:
1. ✅ `time.perf_counter()` for latency measurement
2. ✅ 50/90/95/99th percentile latencies
3. ✅ Interactive Plotly charts
4. ✅ Export to `docs/metrics.html` for GitHub Pages

### Implementation:

#### Add Latency Measurement:

```python
# Cell: Latency Measurement with Percentiles

from utils import LatencyTracker, calculate_latency_stats, plot_latency_histogram
import time

print("⏱️ Measuring Inference Latency...")

# Create tracker
tracker = LatencyTracker()

# Measure latency for multiple samples
n_measurements = 1000
sample_size = 10  # Batch inference

for i in range(n_measurements):
    # Randomly sample features
    sample_idx = np.random.choice(len(X_test), size=sample_size, replace=False)
    X_sample = X_test.iloc[sample_idx]

    # Measure inference time
    with tracker.time():
        _ = calibrated_model.predict_proba(X_sample)

# Calculate statistics
latency_stats = calculate_latency_stats(np.array(tracker.measurements))

print(f"\n📊 Latency Statistics (batch size = {sample_size}):")
print(f"   Mean:   {latency_stats['mean']:.2f} ms")
print(f"   Std:    {latency_stats['std']:.2f} ms")
print(f"   Min:    {latency_stats['min']:.2f} ms")
print(f"   Max:    {latency_stats['max']:.2f} ms")
print(f"\n   P50:    {latency_stats['p50']:.2f} ms")
print(f"   P90:    {latency_stats['p90']:.2f} ms")
print(f"   P95:    {latency_stats['p95']:.2f} ms")
print(f"   P99:    {latency_stats['p99']:.2f} ms")

# Save latency histogram
plot_latency_histogram(
    np.array(tracker.measurements),
    Path("reports/latency_histogram.png"),
    title=f"Inference Latency Distribution (batch={sample_size})"
)
```

#### Convert to Interactive Plotly Visualizations:

```python
# Cell: Interactive Plotly Dashboard

from utils import (
    create_interactive_roc_curve,
    create_interactive_pr_curve,
    create_interactive_confusion_matrix,
    create_interactive_feature_importance,
    create_interactive_calibration_curve,
    create_metrics_dashboard,
    export_to_html
)

print("🎨 Creating Interactive Plotly Dashboard...")

# Individual plots
fig_roc = create_interactive_roc_curve(y_test, y_pred_proba, "ROC Curve - Exoplanet Detection")
fig_pr = create_interactive_pr_curve(y_test, y_pred_proba, "Precision-Recall Curve")
fig_cm = create_interactive_confusion_matrix(y_test, (y_pred_proba > 0.5).astype(int))
fig_fi = create_interactive_feature_importance(
    feature_names=numerical_features,
    importances=pipeline.named_steps['classifier'].feature_importances_,
    top_n=15
)
fig_calib = create_interactive_calibration_curve(
    y_true=y_test,
    predictions={
        "Uncalibrated": y_pred_uncalib,
        "Isotonic": y_pred_isotonic,
        "Platt": y_pred_platt
    }
)

# Comprehensive dashboard
fig_dashboard = create_metrics_dashboard(
    y_true=y_test,
    y_pred_proba=y_pred_proba,
    y_pred=(y_pred_proba > 0.5).astype(int),
    feature_names=numerical_features,
    feature_importances=pipeline.named_steps['classifier'].feature_importances_
)

# Export individual plots
export_to_html(fig_roc, Path("docs/roc_curve.html"))
export_to_html(fig_pr, Path("docs/pr_curve.html"))
export_to_html(fig_fi, Path("docs/feature_importance.html"))
export_to_html(fig_calib, Path("docs/calibration_curves.html"))

# Export comprehensive dashboard
export_to_html(fig_dashboard, Path("docs/metrics.html"))

print("\n✅ Interactive HTML dashboards created:")
print("   📄 docs/metrics.html (comprehensive)")
print("   📄 docs/roc_curve.html")
print("   📄 docs/pr_curve.html")
print("   📄 docs/feature_importance.html")
print("   📄 docs/calibration_curves.html")
print("\n💡 Deploy to GitHub Pages or open in browser!")
```

---

## 📦 Required Dependencies

Add to notebook Cell 2 (after installations):

```python
# Install additional dependencies (if not already installed)
!pip install -q plotly pyyaml kaleido
```

---

## 🧪 TDD Test Coverage

All implementations are tested in:

### tests/test_model_card.py
- ✅ Model card creation with all required fields
- ✅ Save and load functionality
- ✅ Metrics preservation

### tests/test_calibration_curves.py
- ✅ Calibration curve plotting
- ✅ Both Isotonic and Platt methods
- ✅ Brier score comparison

### tests/test_output_schema.py
- ✅ Standardized DataFrame creation
- ✅ All 16+ required columns
- ✅ CSV and JSONL export
- ✅ Sorting by model_score

### tests/test_latency_metrics.py
- ✅ Latency measurement with context manager
- ✅ Percentile calculations (50/90/95/99)
- ✅ Statistics summary
- ✅ Histogram generation

---

## 🚀 Deployment Checklist

### For Notebook 03:
- [ ] Add Platt calibration option
- [ ] Save calibration curves to `reports/`
- [ ] Generate and save Model Card
- [ ] Verify both calibration methods work

### For Notebook 04:
- [ ] Implement standardized CSV export
- [ ] Add JSONL export option
- [ ] Create provenance YAML file
- [ ] Verify all schema columns present

### For Notebook 05:
- [ ] Add latency measurement code
- [ ] Calculate percentiles (50/90/95/99)
- [ ] Convert matplotlib to Plotly
- [ ] Export interactive HTML to `docs/`
- [ ] Test GitHub Pages deployment

---

## 📝 Notes

1. **TDD Philosophy**: Tests were written FIRST (RED phase), then implementations (GREEN phase)
2. **Modularity**: All utilities in `src/utils/` for reusability
3. **Standards**: Follows 2025 best practices for ML model documentation
4. **Reproducibility**: All operations use `random_state=42`
5. **GitHub Pages**: HTML files in `docs/` are ready for deployment

---

## 🔗 Related Files

- `src/utils/model_card.py` - Model Card generation
- `src/utils/provenance.py` - Provenance tracking
- `src/utils/calibration_viz.py` - Calibration visualization
- `src/utils/output_schema.py` - Standardized output
- `src/utils/latency_metrics.py` - Latency measurement
- `src/utils/plotly_viz.py` - Interactive visualizations