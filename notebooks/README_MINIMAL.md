# 03_injection_train_MINIMAL.ipynb - Quick Start Guide

## What This Notebook Does

Trains an exoplanet detection model using NASA data with proper cell execution order.

## Quick Start

### Google Colab
```bash
1. Upload notebook to Colab
2. Run Cell 3 (installation)
3. Click "Restart Runtime"
4. Run Cell 5 → Cell 19
```

### Local Jupyter
```bash
pip install numpy==1.26.4 'scipy<1.13' astropy lightkurve xgboost scikit-learn joblib pandas
jupyter notebook 03_injection_train_MINIMAL.ipynb
# Run Cell 5 → Cell 19 (skip Cell 3)
```

## Cell Execution Order (CRITICAL)

**DO NOT skip cells or run out of order!**

| Cell | Must Run | Why |
|------|----------|-----|
| 5 | ✅ FIRST | Imports all libraries |
| 7 | ✅ SECOND | Defines helper functions |
| 9 | ✅ THIRD | Loads dataset |
| 11 | ✅ FOURTH | Defines feature extraction |
| 13 | ✅ FIFTH | **Extracts features → DEFINES feature_cols** |
| 15 | ✅ SIXTH | Prepares data (USES feature_cols) |
| 17 | ✅ SEVENTH | Trains model (USES feature_cols) |
| 19 | ✅ EIGHTH | Saves model |

## Key Variables Created

| Variable | Cell | Used By |
|----------|------|---------|
| `samples_df` | 9 | 13 (feature extraction) |
| `extract_features_batch()` | 11 | 13 (extraction) |
| `features_df` | 13 | 15 (data prep) |
| **`feature_cols`** | **13** | **15, 17 (CRITICAL!)** |
| `X`, `y`, `groups` | 15 | 17 (training) |
| `best_model` | 17 | 19 (saving) |

## Common Issues

### Error: "name 'feature_cols' is not defined"
**Solution**: Run Cell 13 BEFORE Cell 15

### Error: "samples_df not found"
**Solution**: Run Cell 9 BEFORE Cell 13

### Error: "extract_features_batch not defined"
**Solution**: Run Cell 11 BEFORE Cell 13

## Output Files

All saved to `models/` directory:
- `exoplanet_xgboost_pipeline.pkl` - Trained model
- `feature_columns.txt` - Feature names
- `training_metrics.csv` - Cross-validation results
- `training_summary.txt` - Training overview

## Configuration

### Training Sample Limit
**Cell 13, Line 6:**
```python
features_df = extract_features_batch(samples_df, max_samples=50)
# Change to None for full dataset:
features_df = extract_features_batch(samples_df, max_samples=None)
```

### Cross-Validation Folds
**Cell 17, Line 6:**
```python
n_splits = 5  # Change to 10 for more robust validation
```

### XGBoost Parameters
**Cell 17:**
```python
n_estimators=100,  # More trees = better performance, slower training
max_depth=6,       # Deeper trees = more complex model
learning_rate=0.1  # Lower = slower but more accurate
```

## Runtime Estimates

| Samples | Time |
|---------|------|
| 50 | ~10-15 min |
| 100 | ~20-30 min |
| 500 | ~1-2 hours |
| Full dataset | ~3-6 hours |

## Troubleshooting

### GPU Not Detected
```
ℹ️ No GPU, using tree_method='hist' with CPU
```
**Normal**: Training will use CPU (slower but works)

### Download Failures
```
⚠️ No data for TIC XXXXX sector XX
```
**Normal**: Some light curves may be unavailable, skipped automatically

### Memory Issues
**Reduce `max_samples` in Cell 13:**
```python
features_df = extract_features_batch(samples_df, max_samples=20)
```

## Validation

After training completes, check:
```bash
ls models/
# Should see:
# - exoplanet_xgboost_pipeline.pkl
# - feature_columns.txt
# - training_metrics.csv
# - training_summary.txt
```

## Next Steps

1. **Verify Training**: Check `models/training_summary.txt`
2. **Make Predictions**: Use `04_newdata_inference.ipynb`
3. **Analyze Performance**: Use `05_metrics_dashboard.ipynb`

## Need Help?

**Check these files:**
- `docs/03_MINIMAL_NOTEBOOK_FIX.md` - Detailed fix documentation
- `PROJECT_MEMORY.md` - Full project history
- `CLAUDE.md` - Development guidelines