# 🚀 Quick Start Guide - Notebook Implementation

**Fast Reference for Implementing the Three Notebooks**

---

## 📋 Checklist: Before You Start

- [ ] Review specifications: `docs/NOTEBOOK_SPECIFICATIONS.md`
- [ ] Read architecture: `docs/NOTEBOOK_ARCHITECTURE.md`
- [ ] Check data exists: `data/supervised_dataset.csv` (11,979 rows)
- [ ] Setup Google Colab account
- [ ] Prepare GitHub token (if pushing results)

---

## 🎯 Notebook 02: Feature Extraction

### Quick Facts
- **Time**: 20-30 hours (background execution)
- **Input**: `supervised_dataset.csv` (11,979 samples)
- **Output**: `bls_tls_features.csv` (11,979 × 31)
- **GPU**: Not required

### Implementation Steps

**Step 1**: Create notebook structure (30 minutes)
```python
# Cell 1: Install packages
!pip install numpy==1.26.4 scipy'<1.13' pandas astropy
!pip install lightkurve transitleastsquares matplotlib tqdm

# Cell 2: Imports and config
CONFIG = {
    'BATCH_SIZE': 100,
    'CHECKPOINT_FREQ': 100,
    'PERIOD_MIN': 0.5,
    'PERIOD_MAX': 20.0,
}

# Cell 3: Helper functions (download, preprocess, BLS, TLS, features)
# Cell 4: Checkpoint system (save/load)
# Cell 5: Main processing loop
# Cell 6: Results aggregation
# Cell 7: Visualization
# Cell 8: GitHub push
```

**Step 2**: Test with 10 samples (1 hour)
```python
# Test subset
df_test = df.head(10)
# Run full pipeline
# Verify checkpoint works
# Check output format
```

**Step 3**: Run full dataset (20-30 hours)
```python
# Enable Colab background execution
# Runtime → Manage sessions → Keep session alive
# Start processing
# Monitor checkpoints
```

**Acceptance Criteria**:
- ✅ 10,780+ samples processed (>90% success)
- ✅ Output CSV has 31 columns
- ✅ Checkpoint system tested
- ✅ Processing time <30 hours

---

## 🤖 Notebook 03: Model Training

### Quick Facts
- **Time**: 5-10 minutes (GPU), <30 min (CPU)
- **Input**: `bls_tls_features.csv`
- **Output**: Trained XGBoost model
- **GPU**: Recommended (T4/A100)

### Implementation Steps

**Step 1**: Setup and data loading (15 minutes)
```python
# Cell 1: Install XGBoost with GPU
!pip install xgboost scikit-learn joblib

# Cell 2: Load data
df = pd.read_csv('data/bls_tls_features.csv')
df_success = df[df['success_flag'] == True]

# Cell 3: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

**Step 2**: Train models (30 minutes)
```python
# Cell 4: Logistic Regression (baseline)
lr = LogisticRegressionCV(cv=5)
lr.fit(X_train_scaled, y_train)

# Cell 5: Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

# Cell 6: XGBoost with GPU
xgb = XGBClassifier(
    device='cuda',
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
xgb.fit(X_train_scaled, y_train)
```

**Step 3**: Calibration and evaluation (15 minutes)
```python
# Cell 7: Isotonic calibration
calibrator = IsotonicRegression()
calibrator.fit(xgb.predict_proba(X_val)[:, 1], y_val)

# Cell 8: Evaluation
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC: {roc_auc:.3f}")  # Target: ≥0.92
```

**Step 4**: Save artifacts (5 minutes)
```python
# Cell 9: Persist model
joblib.dump(xgb, 'models/best_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(calibrator, 'models/calibrator.joblib')
```

**Acceptance Criteria**:
- ✅ XGBoost ROC-AUC ≥ 0.92
- ✅ Training time < 15 minutes (GPU)
- ✅ Calibration ECE < 0.10
- ✅ All artifacts saved

---

## 🔮 Notebook 04: Inference Pipeline

### Quick Facts
- **Time**: <60 seconds per target
- **Input**: TIC ID(s)
- **Output**: Predictions with probabilities
- **GPU**: Optional (for batch acceleration)

### Implementation Steps

**Step 1**: Load artifacts (5 minutes)
```python
# Cell 1: Load model and artifacts
model = joblib.load('models/best_model.joblib')
scaler = joblib.load('models/scaler.joblib')
calibrator = joblib.load('models/calibrator.joblib')
```

**Step 2**: Implement single target pipeline (20 minutes)
```python
# Cell 2: Single target inference
def predict_single_target(tic_id):
    # Download light curve
    lc = download_light_curve(tic_id)

    # Preprocess
    lc_clean = preprocess_light_curve(lc)

    # Extract features
    features = extract_all_features(lc_clean)

    # Predict
    features_scaled = scaler.transform([features])
    prob_raw = model.predict_proba(features_scaled)[0, 1]
    prob_calibrated = calibrator.predict([prob_raw])[0]

    return {
        'tic_id': tic_id,
        'probability': prob_calibrated,
        'bls_period': features['bls_period'],
        'bls_snr': features['bls_snr']
    }
```

**Step 3**: Implement batch processing (15 minutes)
```python
# Cell 3: Batch inference
def predict_batch(tic_list):
    results = []
    for tic in tqdm(tic_list):
        try:
            result = predict_single_target(tic)
            results.append(result)
        except Exception as e:
            results.append({
                'tic_id': tic,
                'success': False,
                'error': str(e)
            })

    df = pd.DataFrame(results)
    return df.sort_values('probability', ascending=False)
```

**Step 4**: Add visualizations (20 minutes)
```python
# Cell 4: Visualization function
def visualize_candidate(tic_id, result):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    # 1. Light curve
    # 2. BLS power spectrum
    # 3. Folded light curve
    # 4. Transit zoom
    # 5. Probability bar
    # 6. Feature values
    plt.tight_layout()
    plt.show()
```

**Step 5**: Test with known targets (10 minutes)
```python
# Cell 5: Test with TOI-431
result = predict_single_target('TIC 25155310')
print(f"Probability: {result['probability']:.3f}")
visualize_candidate('TIC 25155310', result)
```

**Acceptance Criteria**:
- ✅ Single target < 60 seconds
- ✅ Batch processing works
- ✅ Visualizations correct
- ✅ Export to CSV functional

---

## 🧪 Testing Strategy

### Unit Testing (Each Notebook)
```python
# Test 1: Data loading
assert df.shape[0] == expected_rows
assert 'tid' in df.columns

# Test 2: Feature extraction (Notebook 02)
features = extract_features(lc, bls, tls, catalog)
assert len(features) == 27
assert not any(np.isnan(list(features.values())))

# Test 3: Model prediction (Notebook 03)
y_pred = model.predict(X_test)
assert len(y_pred) == len(X_test)
assert roc_auc_score(y_test, y_pred) >= 0.92

# Test 4: Inference pipeline (Notebook 04)
result = predict_single_target('TIC 25155310')
assert 'probability' in result
assert 0 <= result['probability'] <= 1
```

### Integration Testing
```python
# Test end-to-end pipeline
# 1. Extract features for 10 samples
# 2. Train model on subset
# 3. Inference on new target
# 4. Verify output format
```

---

## 🚨 Common Issues & Solutions

### Issue 1: NumPy 2.0 Compatibility
**Symptom**: `AttributeError` from lightkurve/TLS
**Solution**:
```python
!pip install numpy==1.26.4 --force-reinstall
# Then restart runtime
```

### Issue 2: Colab Disconnect During Processing
**Symptom**: Processing stops, progress lost
**Solution**:
```python
# Enable background execution
# Use checkpoint system
checkpoint = load_checkpoint()
start_idx = checkpoint['last_index'] if checkpoint else 0
```

### Issue 3: Out of Memory
**Symptom**: Runtime crashes, "Out of memory"
**Solution**:
```python
# Reduce batch size
CONFIG['BATCH_SIZE'] = 50

# Clear memory between batches
import gc
gc.collect()
torch.cuda.empty_cache()
```

### Issue 4: MAST Download Timeout
**Symptom**: `TimeoutError` from lightkurve
**Solution**:
```python
# Increase timeout
search_result = lk.search_lightcurve(
    tid,
    mission='TESS',
    timeout=120  # Increase from default 60
)

# Retry logic
for attempt in range(3):
    try:
        lc = search_result[0].download()
        break
    except:
        if attempt < 2:
            time.sleep(2 ** attempt)
```

### Issue 5: XGBoost GPU Not Found
**Symptom**: "No GPU found, using CPU"
**Solution**:
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")

# If unavailable, use CPU
xgb = XGBClassifier(device='cpu')  # Fallback
```

---

## 📊 Expected Outputs

### Notebook 02 Output
```
data/bls_tls_features.csv:
- 11,979 rows (or >10,780 successful)
- 31 columns (27 features + 4 metadata)
- Size: ~5-10 MB
- Format: CSV with header

checkpoints/:
- bls_checkpoint_XXXXX.pkl files
- Most recent checkpoint used for resume
```

### Notebook 03 Output
```
models/:
- best_model.joblib (XGBoost, ~20-30 MB)
- scaler.joblib (~1 MB)
- calibrator.joblib (~1 MB)
- feature_schema.json (<1 MB)
- training_report.json (<1 MB)

Training metrics:
- ROC-AUC: ≥0.92
- Precision: ≥0.85
- Recall: ≥0.80
- ECE: <0.10
```

### Notebook 04 Output
```
results/candidates_YYYYMMDD.csv:
- Columns: tic_id, probability, bls_period, bls_depth, bls_snr
- Sorted by probability (descending)
- Metadata: model_version, timestamp, etc.

Visualizations:
- Folded light curve plot
- BLS power spectrum
- Probability bar chart
- Feature importance
- Transit shape
- Phase coverage
```

---

## 🎯 Success Metrics

### Notebook 02
| Metric | Target | How to Check |
|--------|--------|--------------|
| **Samples Processed** | >10,780 | `len(df[df['success_flag']==True])` |
| **Success Rate** | >90% | `df['success_flag'].mean()` |
| **Feature Columns** | 27 | `len([c for c in df.columns if c not in metadata])` |
| **Processing Time** | <30 hours | Monitor checkpoints |

### Notebook 03
| Metric | Target | How to Check |
|--------|--------|--------------|
| **ROC-AUC** | ≥0.92 | `roc_auc_score(y_test, y_pred_proba)` |
| **PR-AUC** | ≥0.85 | `average_precision_score(y_test, y_pred_proba)` |
| **Calibration ECE** | <0.10 | Custom ECE function |
| **Training Time** | <15 min (GPU) | Time cells |

### Notebook 04
| Metric | Target | How to Check |
|--------|--------|--------------|
| **Single Target** | <60s | Time `predict_single_target()` |
| **Batch (10)** | <10 min | Time `predict_batch(tic_list)` |
| **Visualization** | 6 plots | Count `plt.subplots()` |
| **Export Format** | Valid CSV | `pd.read_csv(output).shape` |

---

## 📅 Timeline

### Day 1: Notebook 02 Implementation
- Morning: Setup + helper functions (3 hours)
- Afternoon: Test with 10 samples (2 hours)
- Evening: Start full processing (kick off 20-30 hour run)

### Day 2: Notebook 03 Implementation
- Morning: Data loading + preprocessing (2 hours)
- Afternoon: Model training + evaluation (3 hours)
- Evening: Calibration + persistence (2 hours)

### Day 3: Notebook 04 Implementation
- Morning: Load artifacts + single target (2 hours)
- Afternoon: Batch processing + visualization (3 hours)
- Evening: Testing + documentation (2 hours)

### Day 4: Testing & Refinement
- Full end-to-end testing
- Bug fixes and optimizations
- Documentation updates
- GitHub push

**Total**: 3-4 days (including 20-30 hour background execution)

---

## 🔗 Reference Links

### Full Documentation
- [Complete Specifications](./NOTEBOOK_SPECIFICATIONS.md) - Detailed requirements
- [Architecture Diagrams](./NOTEBOOK_ARCHITECTURE.md) - Visual pipeline
- [Summary](./SPECIFICATION_SUMMARY.md) - Executive overview

### External Resources
- [Lightkurve Docs](https://docs.lightkurve.org/)
- [TLS GitHub](https://github.com/hippke/tls)
- [XGBoost GPU](https://xgboost.readthedocs.io/en/stable/gpu/)
- [Colab Guide](https://colab.research.google.com/)

### Project Files
- [Project Memory](../PROJECT_MEMORY.md) - Historical context
- [CLAUDE.md](../CLAUDE.md) - Development guide
- [README.md](../README.md) - Project overview

---

## 💡 Pro Tips

### For Notebook 02
1. **Test checkpoint system first** - Verify save/resume works before long run
2. **Use Google Drive** - Mount Drive for persistent checkpoint storage
3. **Monitor progress** - Check logs every few hours
4. **Start small** - Test with 10, then 100, then full dataset

### For Notebook 03
1. **Use GPU runtime** - Much faster training (5-10 min vs 30 min)
2. **Early stopping** - Prevents overfitting and saves time
3. **Cross-validation** - Don't skip this, ensures robustness
4. **Save everything** - Model, scaler, calibrator, feature names

### For Notebook 04
1. **Test with known targets** - Use TOI-431, TOI-270, etc.
2. **Error handling** - Some downloads will fail, handle gracefully
3. **Batch processing** - More efficient than sequential
4. **Cache results** - Don't re-download if already processed

---

**Status**: ✅ READY TO IMPLEMENT
**Next Action**: Start Notebook 02 implementation
**Questions?**: Check [NOTEBOOK_SPECIFICATIONS.md](./NOTEBOOK_SPECIFICATIONS.md)

---

*Quick Start Guide for NASA Exoplanet Detection Pipeline*
*Last updated: 2025-09-30*