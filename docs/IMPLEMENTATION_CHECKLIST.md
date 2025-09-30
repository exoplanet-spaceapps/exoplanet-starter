# ✅ Implementation Checklist - Colab Exoplanet Pipeline

**Version**: 1.0
**Date**: 2025-09-30
**Status**: Ready to Execute

---

## 📋 Pre-Implementation Setup

### Environment Verification
- [ ] Google Colab account accessible
- [ ] GitHub repository with `supervised_dataset.csv` (11,979 samples)
- [ ] Colab Pro subscription (recommended for GPU access)
- [ ] GitHub personal access token (for pushing results)

### Documentation Review
- [ ] Read `COLAB_IMPLEMENTATION_PLAN.md` (comprehensive plan)
- [ ] Review `ARCHITECTURE_DIAGRAM.txt` (system overview)
- [ ] Check `QUICK_REFERENCE.md` (one-page cheat sheet)
- [ ] Study `TDD_TEST_SPECIFICATIONS.md` (test strategy)

---

## 🔬 Phase 3: Feature Extraction (02_bls_baseline.ipynb)

### Setup Tasks (Est. 10 minutes)
- [ ] Open notebook in Google Colab
- [ ] Run Cell 1: Install dependencies
  ```python
  !pip install -q numpy==1.26.4 scipy'<1.13' astropy lightkurve transitleastsquares wotan
  ```
- [ ] **CRITICAL**: Restart runtime after NumPy install
- [ ] Run Cell 2: Import libraries and configure environment
- [ ] Verify `supervised_dataset.csv` is accessible (11,979 rows)

### TDD Implementation (Est. 1 hour)
- [ ] Copy test functions from `TDD_TEST_SPECIFICATIONS.md`
- [ ] Run `test_bls_period_recovery_simple()` → should FAIL initially
- [ ] Implement `run_bls()` function from `app/bls_features.py`
- [ ] Run test again → should PASS
- [ ] Repeat for `test_feature_extraction_completeness()`
- [ ] Repeat for `test_checkpoint_save_load()`
- [ ] All Phase 3 tests passing before proceeding

### Feature Extraction Execution (Est. 6-8 hours)
- [ ] Configure batch processing parameters:
  ```python
  BATCH_SIZE = 50
  CHECKPOINT_INTERVAL = 100
  MAX_WORKERS = 8  # CPU cores
  ```
- [ ] Start main extraction loop
- [ ] Monitor progress (checkpoints saved every 100 samples)
- [ ] Handle errors gracefully (log failed targets, continue)
- [ ] Verify checkpoint files created in `data/features/`

### Quality Validation (Est. 15 minutes)
- [ ] Load final `bls_tls_features.csv`
- [ ] Check row count (expect ~10,182 successful extractions)
- [ ] Verify 14 feature columns present
- [ ] Check for NaN values in critical features (`bls_period`, `bls_depth`, `bls_snr`)
- [ ] Inspect sample features (head/tail/describe)
- [ ] Generate `feature_extraction_report.json`

### Deliverables Checklist
- [ ] `bls_tls_features.csv` (5 MB, ~10,182 rows × 16 columns)
- [ ] `feature_extraction_report.json` (success/failure statistics)
- [ ] Checkpoint files (temporary, can delete after success)
- [ ] Optional: Push results to GitHub

### Success Criteria ✅
- [ ] ≥85% success rate (≥10,182 of 11,979 targets)
- [ ] All 14 features extracted
- [ ] No NaN in critical features
- [ ] Execution completed within 8 hours
- [ ] All Phase 3 tests passing

---

## 🤖 Phase 4: Model Training (03_injection_train.ipynb)

### Setup Tasks (Est. 5 minutes)
- [ ] Open notebook in Google Colab
- [ ] Enable GPU: Runtime → Change runtime type → T4 GPU
- [ ] Verify GPU detected:
  ```python
  import torch
  print(torch.cuda.is_available())  # Should be True
  print(torch.cuda.get_device_name(0))  # Should show "Tesla T4" or "L4"
  ```
- [ ] Install XGBoost:
  ```python
  !pip install -q xgboost scikit-learn imbalanced-learn
  ```
- [ ] Upload or load `bls_tls_features.csv` from Phase 3

### TDD Implementation (Est. 30 minutes)
- [ ] Copy Phase 4 test functions
- [ ] Run `test_gpu_detection()` → should PASS if GPU available
- [ ] Run `test_calibration_improves_brier_score()` → should FAIL initially
- [ ] Implement calibration pipeline
- [ ] Run test again → should PASS
- [ ] Run `test_cross_validation_consistency()` → verify consistent performance
- [ ] All Phase 4 tests passing

### Data Preprocessing (Est. 5 minutes)
- [ ] Load features: `df = pd.read_csv('data/bls_tls_features.csv')`
- [ ] Check dataset shape: `(~10182, 16)`
- [ ] Handle missing values (imputation or drop)
- [ ] Separate features (X) and labels (y)
- [ ] Train/val/test split: 70/15/15
- [ ] StandardScaler fit on training data
- [ ] Verify class balance: ~50/50 positive/negative

### XGBoost Training (Est. 15-30 minutes on GPU)
- [ ] Configure XGBoost parameters:
  ```python
  params = {
      'tree_method': 'gpu_hist',
      'gpu_id': 0,
      'max_depth': 8,
      'learning_rate': 0.05,
      'n_estimators': 500,
      'early_stopping_rounds': 50,
      'eval_metric': ['auc', 'logloss']
  }
  ```
- [ ] Create DMatrix for GPU training
- [ ] Train with early stopping (monitor validation AUC)
- [ ] Note final number of trees (likely 300-400)
- [ ] Training log shows GPU utilization

### Cross-Validation (Est. 10 minutes)
- [ ] 5-fold Stratified CV
- [ ] Record scores for each fold
- [ ] Compute mean and std dev
- [ ] Verify consistency (std < 0.1)
- [ ] Expected: ROC-AUC ~0.950 ± 0.002

### Probability Calibration (Est. 5 minutes)
- [ ] Use separate calibration set (15% of data)
- [ ] Apply isotonic regression: `CalibratedClassifierCV(method='isotonic')`
- [ ] Compute Brier score before/after
- [ ] Compute ECE (Expected Calibration Error)
- [ ] Expected improvements:
  - Brier: 0.095 → 0.082
  - ECE: 0.051 → 0.034

### Model Evaluation (Est. 10 minutes)
- [ ] Test set predictions (15% holdout)
- [ ] Compute metrics:
  - [ ] ROC-AUC ≥0.92
  - [ ] PR-AUC ≥0.90
  - [ ] Precision @ 0.5 ≥0.90
  - [ ] Recall @ 0.5 ≥0.90
  - [ ] F1-Score ≥0.90
  - [ ] Brier Score <0.09
  - [ ] ECE <0.05
- [ ] Generate visualizations:
  - [ ] ROC curve with 95% CI
  - [ ] Precision-Recall curve
  - [ ] Calibration curve (reliability diagram)
  - [ ] Feature importance plot

### Model Persistence (Est. 2 minutes)
- [ ] Save calibrated model: `joblib.dump(calibrated_model, 'xgboost_calibrated.joblib')`
- [ ] Save scaler: `joblib.dump(scaler, 'scaler.joblib')`
- [ ] Save feature schema: `feature_schema.json`
- [ ] Generate training report: `training_report.json`
- [ ] Download all artifacts to local machine

### Deliverables Checklist
- [ ] `xgboost_calibrated.joblib` (5 MB)
- [ ] `scaler.joblib` (1 KB)
- [ ] `feature_schema.json`
- [ ] `training_report.json` (with all metrics)
- [ ] Visualization PNGs (4 files)
- [ ] Cross-validation results CSV

### Success Criteria ✅
- [ ] ROC-AUC ≥0.92 on test set
- [ ] PR-AUC ≥0.90 on test set
- [ ] ECE <0.05 (good calibration)
- [ ] Training completed in <60 minutes (GPU)
- [ ] All Phase 4 tests passing
- [ ] Model files successfully saved

---

## 🔮 Phase 5: Inference Pipeline (04_newdata_inference.ipynb)

### Setup Tasks (Est. 3 minutes)
- [ ] Open notebook in Google Colab
- [ ] Install dependencies:
  ```python
  !pip install -q lightkurve transitleastsquares joblib
  ```
- [ ] Upload model artifacts:
  - [ ] `xgboost_calibrated.joblib`
  - [ ] `scaler.joblib`
  - [ ] `feature_schema.json`
- [ ] Load model and scaler

### TDD Implementation (Est. 20 minutes)
- [ ] Copy Phase 5 test functions
- [ ] Run `test_inference_handles_failures()` → should FAIL initially
- [ ] Implement error handling in inference pipeline
- [ ] Run test again → should PASS
- [ ] Run `test_ranking_prioritizes_high_confidence()`
- [ ] Implement ranking logic
- [ ] All Phase 5 tests passing

### User Input Configuration (Est. 2 minutes)
- [ ] Define TIC IDs to analyze:
  ```python
  TIC_IDS = [
      'TIC 25155310',    # Known planet (TOI 270 b)
      'TIC 307210830',   # User target 1
      'TIC 141527766',   # User target 2
      # Add more...
  ]
  ```
- [ ] Set parallel workers: `N_WORKERS = 4`
- [ ] Configure filters:
  - Minimum probability: 0.70
  - Minimum SNR: 7.0

### Inference Execution (Est. 5-10 min per target)
- [ ] Start batch inference
- [ ] Monitor download progress
- [ ] Handle failed downloads gracefully
- [ ] Extract features for each target
- [ ] Run model predictions
- [ ] Collect results in DataFrame

### Results Analysis (Est. 5 minutes)
- [ ] Apply ranking algorithm
- [ ] Filter by thresholds
- [ ] Sort by composite score
- [ ] Validate known planets ranked highly
- [ ] Inspect top 10 candidates
- [ ] Check for suspicious false positives

### Output Generation (Est. 2 minutes)
- [ ] Create `candidates_YYYYMMDD.csv`:
  ```csv
  tic_id,planet_probability,bls_period,bls_depth,bls_snr,rank
  TIC 25155310,0.952,3.36,0.0082,28.5,1
  ```
- [ ] Generate `inference_report.json`
- [ ] Save visualizations (optional):
  - Top candidates light curves
  - BLS periodograms
- [ ] Download results

### Validation Tests (Est. 5 minutes)
- [ ] Verify known planet (TOI 270 b) has high probability (>0.8)
- [ ] Check that failed targets are properly logged
- [ ] Ensure all input TICs have output records
- [ ] Validate CSV format and columns

### Deliverables Checklist
- [ ] `candidates_YYYYMMDD.csv` (primary output)
- [ ] `inference_report.json` (execution summary)
- [ ] Optional: Light curve plots for top candidates
- [ ] Optional: BLS periodogram plots

### Success Criteria ✅
- [ ] Known planets ranked in top 10%
- [ ] Inference completes without crashes
- [ ] All input TICs processed (success or logged failure)
- [ ] Output CSV properly formatted
- [ ] All Phase 5 tests passing
- [ ] Ranking logic prioritizes high-confidence targets

---

## 🎯 Final Integration & Validation

### End-to-End Validation (Est. 15 minutes)
- [ ] Verify data flow: Raw data → Features → Model → Predictions
- [ ] Check file sizes are reasonable:
  - Features CSV: ~5 MB
  - Model: ~5 MB
  - Candidates CSV: <1 MB
- [ ] Validate reproducibility:
  - Re-run Phase 5 with same TICs
  - Results should be identical
- [ ] Test on multiple known planets
- [ ] Compare with NASA Archive labels

### Documentation Updates (Est. 30 minutes)
- [ ] Update `PROJECT_MEMORY.md` with:
  - Phase 3-5 completion status
  - Final metrics achieved
  - Any issues encountered and solutions
- [ ] Update `README.md` with:
  - Final performance numbers
  - Link to trained model
  - Example inference results
- [ ] Create `PHASE_3_5_COMPLETION_REPORT.md`:
  - Summary of all phases
  - Final metrics
  - Lessons learned

### Performance Summary (Fill in after completion)
```
Phase 3 (Feature Extraction):
  - Targets processed: _____ / 11,979 (_____%)
  - Execution time: _____ hours
  - Output size: _____ MB

Phase 4 (Model Training):
  - ROC-AUC: _____
  - PR-AUC: _____
  - ECE: _____
  - Training time: _____ minutes (GPU)

Phase 5 (Inference):
  - Targets analyzed: _____
  - Known planets detected: _____ / _____
  - Execution time: _____ minutes
```

### GitHub Integration (Optional, Est. 10 minutes)
- [ ] Push feature CSV to GitHub (via Git LFS)
- [ ] Push model artifacts to GitHub
- [ ] Push results to GitHub
- [ ] Update repository README with badges
- [ ] Create release tag: `v1.0.0-full-pipeline`

---

## 🚨 Common Issues & Solutions

### Issue: Colab disconnects after 6 hours
**Solution**:
- Save checkpoints every 100 samples
- Resume from last checkpoint
- Consider Colab Pro for longer runtimes

### Issue: GPU not detected
**Solution**:
1. Runtime → Change runtime type → T4 GPU
2. Restart runtime
3. Verify with `torch.cuda.is_available()`
4. Fallback to CPU if needed (slower)

### Issue: NumPy compatibility errors
**Solution**:
```python
!pip install -q numpy==1.26.4 --force-reinstall
# MUST restart runtime after this
```

### Issue: MAST download failures
**Solution**:
- Implement retry logic with exponential backoff
- Try alternative authors (SPOC → QLP)
- Skip problematic targets, log errors
- Expect ~15% failure rate

### Issue: Memory errors
**Solution**:
- Reduce batch size (50 → 25)
- Force garbage collection after each batch
- Clear output cells periodically
- Use Colab High-RAM runtime if available

### Issue: Calibration fails
**Solution**:
- Ensure separate calibration set (not used in training)
- Check for class imbalance in calibration set
- Try `method='sigmoid'` if isotonic fails
- Verify sufficient samples (need >100)

---

## 📊 Progress Tracking

### Daily Progress Log

**Day 1** (Est. 8 hours):
- [ ] Morning: Phase 3 setup & TDD (2 hours)
- [ ] Afternoon: Feature extraction execution (6 hours)
- [ ] Evening: Quality validation & checkpoint

**Day 2** (Est. 2 hours):
- [ ] Morning: Phase 4 setup & training (1.5 hours)
- [ ] Afternoon: Model evaluation & save (30 min)

**Day 3** (Est. 1 hour):
- [ ] Morning: Phase 5 inference on test targets (45 min)
- [ ] Afternoon: Documentation & wrap-up (15 min)

### Milestone Markers
- [ ] 🎯 Milestone 1: Phase 3 TDD tests all passing
- [ ] 🎯 Milestone 2: 5,000 targets processed (checkpoint_0050)
- [ ] 🎯 Milestone 3: Feature extraction complete
- [ ] 🎯 Milestone 4: Model achieves ROC-AUC >0.92
- [ ] 🎯 Milestone 5: Model calibrated (ECE <0.05)
- [ ] 🎯 Milestone 6: Known planet correctly classified
- [ ] 🎯 Milestone 7: All phases complete ✅

---

## 🎉 Completion Checklist

### Final Verification
- [ ] All 3 notebooks executed successfully
- [ ] All TDD tests passing (100%)
- [ ] All deliverables generated and downloaded
- [ ] Documentation updated
- [ ] Performance metrics meet success criteria
- [ ] No critical errors or warnings

### Archive & Backup
- [ ] Download all output files to local machine
- [ ] Backup to Google Drive (optional)
- [ ] Push to GitHub (optional)
- [ ] Create project archive ZIP

### Celebration 🎊
- [ ] Review final metrics
- [ ] Compare with project goals
- [ ] Document lessons learned
- [ ] Share results with team
- [ ] Plan next steps (if any)

---

**Last Updated**: 2025-09-30
**Status**: Ready to Execute
**Estimated Total Time**: 11-12 hours (across 3 days)

---

## 📞 Quick Start Command

To begin implementation immediately:

```bash
# 1. Open first notebook
https://colab.research.google.com/github/YOUR_USERNAME/exoplanet-starter/blob/main/notebooks/02_bls_baseline.ipynb

# 2. Run first cell (NumPy fix)
!pip install -q numpy==1.26.4 scipy'<1.13'

# 3. Restart runtime (CRITICAL!)

# 4. Continue with remaining cells
```

**Good luck! 🚀**