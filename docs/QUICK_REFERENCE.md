# 🚀 Quick Reference Card - Colab Exoplanet Pipeline

## 📋 One-Page Cheat Sheet

### Dataset Overview
```
Total Samples: 11,979
├─ Positive: 5,944 (49.6%)
├─ Negative: 6,035 (50.4%)
├─ Source: NASA TOI + KOI False Positives
└─ File: data/supervised_dataset.csv
```

---

## 🎯 Three-Phase Execution

### Phase 3: Feature Extraction (02_bls_baseline.ipynb)
```python
# ⏱️ Time: 6-8 hours | 💾 Output: 5 MB | 🎯 Target: 10,182 samples

# Setup
!pip install -q numpy==1.26.4 lightkurve transitleastsquares

# Load data
df = pd.read_csv('data/supervised_dataset.csv')

# Process with checkpoints
for batch in process_batches(df, batch_size=50):
    features = extract_bls_tls_features(batch)
    if batch_id % 20 == 0:
        save_checkpoint(features, batch_id)

# Output
features.to_csv('data/bls_tls_features.csv')
```

**Key Features (14 total):**
- BLS: period, depth, duration, SNR, t0
- Geometric: duration/period, depth/SNR
- Diagnostic: odd-even diff, symmetry
- Statistical: std, MAD, skew, kurtosis, periodicity

---

### Phase 4: Model Training (03_injection_train.ipynb)
```python
# ⏱️ Time: 30-60 min (GPU) | 💾 Output: 5 MB | 🎯 Target: ROC-AUC > 0.92

# GPU Setup
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")  # T4 or L4

# Train XGBoost
params = {
    'tree_method': 'gpu_hist',      # GPU acceleration
    'max_depth': 8,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'early_stopping_rounds': 50
}

model = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dval, 'val')])

# Calibrate
calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
calibrated.fit(X_cal, y_cal)

# Save
joblib.dump(calibrated, 'model/xgboost_calibrated.joblib')
```

**Expected Metrics:**
- ROC-AUC: 0.950-0.955
- PR-AUC: 0.945-0.950
- ECE (calibration): < 0.05
- Brier Score: < 0.09

---

### Phase 5: Inference (04_newdata_inference.ipynb)
```python
# ⏱️ Time: 5-10 min/target | 💾 Output: 100 KB | 🎯 Target: Ranked candidates

# Load model
model = joblib.load('model/xgboost_calibrated.joblib')
scaler = joblib.load('model/scaler.joblib')

# Input TIC IDs
TIC_IDS = ['TIC 25155310', 'TIC 307210830', 'TIC 141527766']

# Run inference
results = []
for tic_id in TIC_IDS:
    lc = download_lightcurve(tic_id)
    features = extract_features(lc)
    proba = model.predict_proba(scaler.transform([features]))[0, 1]
    results.append({'tic_id': tic_id, 'probability': proba})

# Rank and save
df = pd.DataFrame(results).sort_values('probability', ascending=False)
df.to_csv(f'candidates_{datetime.now():%Y%m%d}.csv')
```

**Output Format:**
```csv
tic_id,planet_probability,bls_period,bls_depth,bls_snr,rank
TIC 25155310,0.952,3.36,0.0082,28.5,1
TIC 307210830,0.885,5.72,0.0045,22.1,2
```

---

## ⚡ Quick Commands

### Colab Setup (All Notebooks)
```python
# NumPy compatibility fix (CRITICAL!)
!pip install -q numpy==1.26.4 scipy'<1.13'

# Restart runtime after NumPy install
# Runtime → Restart runtime
```

### GPU Check (Phase 4 only)
```python
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ No GPU - will use CPU (slower)")
```

### Google Drive Mount (Optional)
```python
from google.colab import drive
drive.mount('/content/drive')

# Save to Drive
output_dir = '/content/drive/MyDrive/exoplanet-results/'
```

---

## 🛠️ Troubleshooting

### Problem: "No data available for TIC"
**Solution:** Try different authors
```python
search = lk.search_lightcurve(tic_id, author='SPOC')  # First try
if len(search) == 0:
    search = lk.search_lightcurve(tic_id, author='QLP')  # Fallback
```

### Problem: "Colab timeout after 6 hours"
**Solution:** Use checkpoints
```python
# Save every 100 samples
if sample_id % 100 == 0:
    features_df.to_parquet(f'checkpoint_{sample_id}.parquet')

# Resume on next run
if Path('checkpoint_5000.parquet').exists():
    features_df = pd.read_parquet('checkpoint_5000.parquet')
    start_idx = 5000  # Resume from here
```

### Problem: "GPU not detected"
**Solution:** Change runtime type
```
Runtime → Change runtime type → Hardware accelerator → T4 GPU
```

### Problem: "Calibration failing"
**Solution:** Check data split
```python
# Need separate calibration set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Train on X_train, calibrate on X_cal, test on X_test
```

---

## 📊 Performance Benchmarks

| Phase | CPU Time | GPU Time | Success Rate | Output Size |
|-------|----------|----------|--------------|-------------|
| Phase 3 | 6-8 hrs | N/A | 85% | 5 MB |
| Phase 4 | 2-3 hrs | 30-60 min | 100% | 5 MB |
| Phase 5 | 5-10 min/target | N/A | ~90% | <1 MB |

---

## 🎯 Success Criteria Checklist

### Phase 3 ✅
- [ ] Processed ≥10,000 targets (85% of 11,979)
- [ ] Generated `bls_tls_features.csv` with 14 features
- [ ] No NaN values in critical features
- [ ] Execution completed within 8 hours

### Phase 4 ✅
- [ ] ROC-AUC ≥0.92 on test set
- [ ] PR-AUC ≥0.90 on test set
- [ ] ECE (calibration error) <0.05
- [ ] Model saved successfully

### Phase 5 ✅
- [ ] Known planets ranked in top 10%
- [ ] Inference completes for new targets
- [ ] CSV output generated
- [ ] No critical errors

---

## 🔗 Quick Links

- **Main Plan**: `docs/COLAB_IMPLEMENTATION_PLAN.md`
- **Architecture**: `docs/ARCHITECTURE_DIAGRAM.txt`
- **Project Memory**: `PROJECT_MEMORY.md`
- **Troubleshooting**: `COLAB_TROUBLESHOOTING.md`

---

## 📞 Next Steps

1. **Review** this guide and main implementation plan
2. **Open** `02_bls_baseline.ipynb` in Colab
3. **Execute** Phase 3 (6-8 hours with checkpoints)
4. **Validate** features before proceeding to Phase 4
5. **Train** model with GPU acceleration
6. **Deploy** inference pipeline

---

**Last Updated**: 2025-09-30
**Status**: Ready for Implementation