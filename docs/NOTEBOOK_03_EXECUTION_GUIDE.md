# Notebook 03 Execution Guide

## Quick Start

### For Google Colab
1. Upload `03_injection_train_FIXED.ipynb` to Colab
2. Clone repository: `!git clone <repo> /content/exoplanet-starter`
3. Run cells 0-4 (setup)
4. Verify imports successful
5. Run remaining cells sequentially

### For Local Jupyter
1. Open notebook: `jupyter notebook 03_injection_train_FIXED.ipynb`
2. Run cells 0-4 (setup)
3. Verify environment detected
4. Continue execution

## Prerequisites

### Required Data Files
- `data/candidates.csv` (from Notebook 01)
- `data/processed/bls_features.csv` (from Notebook 02)

### Required Modules
- `app/bls_features.py`
- `app/injection.py`

### Required Packages
- numpy, pandas, sklearn, xgboost, lightkurve
- matplotlib, seaborn, joblib

## Cell Guide

### Cells 0-4: Setup
**Cell 0** [Markdown]: Overview
**Cell 1** [Code]: Environment check (Colab vs local)
**Cell 2** [Code]: Import all libraries
**Cell 3** [Code]: Setup paths and sys.path
**Cell 4** [Code]: Import project modules

Expected output after Cell 4:
```
All imports completed successfully
Project modules loaded
```

### Cells 5+: Processing
- Data loading
- Feature engineering
- Model training
- Evaluation

## Common Issues

### Import Error
**Problem**: `ModuleNotFoundError`
**Solution**: `pip install <missing-package>`

### Data Not Found
**Problem**: `FileNotFoundError: data/candidates.csv`
**Solution**: Run Notebooks 01 and 02 first

### App Module Not Found
**Problem**: `No module named 'app'`
**Solution**: 
- Check `app/` directory exists
- Create `app/__init__.py` if missing
- Verify PROJECT_ROOT set correctly

### NumPy 2.0 Issues
**Problem**: `AttributeError: module 'numpy' has no attribute 'float'`
**Solution**: `pip install "numpy<2.0"`

## Expected Runtime
- Setup: <1 minute
- Full execution (Colab GPU): 15-30 minutes
- Full execution (CPU): 30-90 minutes

## Success Criteria
- All setup cells run without errors
- Data loads successfully
- Models train and save
- Metrics calculated
- Plots generated

## Output Files
```
models/
  - xgb_model.json
  - calibrated_model.pkl
  - scaler.pkl

results/
  - metrics.json
  - feature_importance.csv
  - plots (PNG files)
```

## Troubleshooting

If execution fails:
1. Check all prerequisites met
2. Verify data files exist
3. Ensure app modules available
4. Check Python package versions
5. Try reducing data size for testing

## Next Steps
1. Review results in `results/`
2. Check model performance metrics
3. Run Notebook 04 (inference) if available
4. Iterate with different parameters

---
Guide for: 03_injection_train_FIXED.ipynb
Version: 1.0
Date: 2025-09-30
