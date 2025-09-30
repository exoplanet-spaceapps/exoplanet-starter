# 📋 Notebook Specifications - Executive Summary

**Version**: 1.0
**Date**: 2025-09-30
**Status**: ✅ Specifications Complete, Ready for Implementation

---

## Quick Reference

### Notebook 02: BLS/TLS Feature Extraction

**Purpose**: Extract astronomical features from 11,979 light curves

**Key Stats**:
- Input: `supervised_dataset.csv` (11,979 samples)
- Output: `bls_tls_features.csv` (27 features + metadata)
- Processing Time: 20-30 hours (with checkpoints)
- Success Target: >90% samples
- Execution: Google Colab with background mode

**Critical Features**:
- ✅ Checkpoint system (resume from interruption)
- ✅ Batch processing (100 samples/batch)
- ✅ Error recovery (continue on failures)
- ✅ NumPy 1.26.4 compatibility fix
- ✅ Memory management (< 8GB peak)

**Acceptance Criteria**:
- [ ] All 11,979 samples processed
- [ ] Success rate > 90%
- [ ] Output has 27 feature columns
- [ ] Checkpoint system tested and working
- [ ] GitHub push successful

---

### Notebook 03: ML Training with GPU

**Purpose**: Train XGBoost classifier on extracted features

**Key Stats**:
- Input: `bls_tls_features.csv`
- Output: Trained XGBoost model + artifacts
- Training Time: 5-10 minutes (GPU), <30 minutes (CPU)
- Target ROC-AUC: ≥ 0.92
- Execution: Google Colab with T4/A100 GPU

**Models Trained**:
1. **Logistic Regression** (baseline): ROC-AUC >0.85, <2min
2. **Random Forest** (ensemble): ROC-AUC >0.90, <5min
3. **XGBoost** (best): ROC-AUC ≥0.92, <10min GPU

**Key Features**:
- ✅ GPU acceleration (`device='cuda'`)
- ✅ 5-fold StratifiedKFold CV
- ✅ Isotonic calibration (ECE <0.10)
- ✅ Early stopping (20 rounds)
- ✅ Full pipeline persistence

**Acceptance Criteria**:
- [ ] XGBoost ROC-AUC ≥ 0.92
- [ ] Training time < 15 minutes with GPU
- [ ] Calibration ECE < 0.10
- [ ] Model exports < 50 MB
- [ ] All artifacts saved correctly

---

### Notebook 04: Inference Pipeline

**Purpose**: Predict exoplanet candidates from new TIC IDs

**Key Stats**:
- Input: TIC ID(s)
- Output: Probability + period + depth + SNR
- Inference Time: <60s single, <10min batch (10 targets)
- Execution: Google Colab (GPU optional)

**Pipeline Stages**:
1. Load model artifacts (<5s)
2. Download light curve (<20s)
3. Preprocess + extract features (<30s)
4. Inference + calibration (<5s)
5. Visualization (<5s)

**Key Features**:
- ✅ Single target inference
- ✅ Batch processing
- ✅ GPU optimization (BFloat16)
- ✅ 6 diagnostic plots
- ✅ Standardized CSV export
- ✅ Error handling

**Acceptance Criteria**:
- [ ] Single target < 60 seconds
- [ ] Batch (10 targets) < 10 minutes
- [ ] Visualizations generated correctly
- [ ] Export matches schema
- [ ] Error handling tested

---

## Implementation Roadmap

### Phase 1: Feature Extraction (Priority 1) ⚡
**Notebook**: 02_bls_baseline.ipynb
**Timeline**: 1-2 days dev + 20-30 hours execution
**Blockers**: None (ready to start)

**Tasks**:
1. Implement checkpoint system
2. Add batch processing logic
3. Integrate BLS/TLS algorithms
4. Test with 100 samples
5. Run full dataset (background)
6. Push results to GitHub

**Deliverable**: `data/bls_tls_features.csv` (11,979 rows)

---

### Phase 2: Model Training (Priority 2) 🎯
**Notebook**: 03_injection_train.ipynb
**Timeline**: 1 day
**Blockers**: Requires Phase 1 completion

**Tasks**:
1. Load feature dataset
2. Implement LogReg baseline
3. Train Random Forest
4. Train XGBoost with GPU
5. Apply calibration
6. Save best model
7. Generate training report

**Deliverable**: Trained XGBoost model (ROC-AUC ≥ 0.92)

---

### Phase 3: Inference Pipeline (Priority 3) 🔮
**Notebook**: 04_newdata_inference.ipynb
**Timeline**: 0.5 days
**Blockers**: Requires Phase 2 completion

**Tasks**:
1. Load model artifacts
2. Implement single target pipeline
3. Implement batch processing
4. Add visualizations
5. Test with known TOIs
6. Document usage

**Deliverable**: Production-ready inference notebook

---

## Data Flow

```
Phase 1: Feature Extraction
├─ Input: supervised_dataset.csv (11,979 samples)
├─ Process: BLS/TLS analysis + feature extraction
└─ Output: bls_tls_features.csv (11,979 × 31)
            ↓
Phase 2: Model Training
├─ Input: bls_tls_features.csv
├─ Process: Train 3 models + calibration
└─ Output: models/best_model.joblib + artifacts
            ↓
Phase 3: Inference
├─ Input: New TIC ID(s) + trained model
├─ Process: Download → Extract → Predict
└─ Output: Predictions with probabilities
```

---

## Key Technical Specifications

### Environment Requirements

**Google Colab Runtime**:
- Type: Standard or GPU (T4/A100 recommended)
- Python: 3.10+
- RAM: 12GB minimum
- Disk: 15GB available

**Package Versions**:
```yaml
critical:
  numpy: 1.26.4  # NOT 2.0+
  scipy: <1.13
  lightkurve: >=2.4.0
  transitleastsquares: >=1.0.31
  xgboost: >=2.0.0
  scikit-learn: >=1.3.0
```

### Performance Targets

| Metric | Notebook 02 | Notebook 03 | Notebook 04 |
|--------|-------------|-------------|-------------|
| **Execution Time** | 20-30 hours | 5-10 min (GPU) | <60s single |
| **Memory Usage** | <8GB peak | <16GB peak | <4GB |
| **Success Rate** | >90% | 100% | >90% |
| **Output Quality** | 27 features | ROC-AUC≥0.92 | Valid predictions |

### Quality Metrics

**Notebook 02**:
- ✅ Feature extraction success rate: >90%
- ✅ Average time per sample: <50 seconds
- ✅ Checkpoint recovery: 100% success
- ✅ Output CSV validation: Pass schema

**Notebook 03**:
- ✅ XGBoost ROC-AUC: ≥0.92
- ✅ Precision @ 80% recall: ≥0.85
- ✅ Calibration ECE: <0.10
- ✅ Training reproducibility: Seed=42

**Notebook 04**:
- ✅ Single target latency: <60s
- ✅ Batch throughput: <1min per target
- ✅ Prediction accuracy: Match validation set
- ✅ Error handling: Continue on failures

---

## Risk Assessment

### Critical Risks 🔴

**Risk 1: Colab Disconnects During 20-Hour Processing**
- **Impact**: High (lose all progress)
- **Mitigation**: Checkpoint every 100 samples + Google Drive persistence
- **Status**: ✅ Mitigated in spec

**Risk 2: NumPy 2.0 Incompatibility**
- **Impact**: Critical (notebooks won't run)
- **Mitigation**: Force NumPy 1.26.4 + runtime restart
- **Status**: ✅ Mitigated in spec

### Medium Risks 🟡

**Risk 3: MAST Download Failures**
- **Impact**: Medium (some samples fail)
- **Mitigation**: Retry logic + continue on failure
- **Status**: ✅ Mitigated in spec

**Risk 4: GPU Unavailable**
- **Impact**: Medium (slower training)
- **Mitigation**: CPU fallback + optimized parameters
- **Status**: ✅ Mitigated in spec

### Low Risks 🟢

**Risk 5: Model Overfitting**
- **Impact**: Low (detected early)
- **Mitigation**: Cross-validation + early stopping
- **Status**: ✅ Mitigated in spec

---

## Success Criteria Checklist

### Overall Project
- [x] ✅ Specifications complete (this document)
- [ ] Notebook 02 implemented and tested
- [ ] Notebook 03 implemented and tested
- [ ] Notebook 04 implemented and tested
- [ ] All notebooks run in fresh Colab
- [ ] Results pushed to GitHub

### Notebook 02: Feature Extraction
- [ ] All 11,979 samples processed
- [ ] Success rate >90%
- [ ] Output CSV has 27 feature columns
- [ ] Checkpoint system works
- [ ] Processing time <30 hours
- [ ] GitHub push successful

### Notebook 03: Model Training
- [ ] XGBoost ROC-AUC ≥0.92
- [ ] Training time <15 minutes (GPU)
- [ ] Calibration ECE <0.10
- [ ] Model exports <50MB
- [ ] Training report generated

### Notebook 04: Inference
- [ ] Single target <60 seconds
- [ ] Batch (10) <10 minutes
- [ ] Visualizations correct
- [ ] Export matches schema
- [ ] Error handling tested

---

## Next Steps

### Immediate Actions (Today)
1. ✅ Review this specification document
2. Create GitHub issue tracking notebooks
3. Setup Colab notebooks from templates
4. Test Notebook 02 with 10 samples

### This Week
1. Implement Notebook 02 feature extraction
2. Test checkpoint system thoroughly
3. Run full dataset (background, 20-30 hours)
4. Begin Notebook 03 development

### Next Week
1. Complete Notebook 03 training
2. Validate model performance
3. Implement Notebook 04 inference
4. End-to-end testing

---

## Documentation Structure

```
docs/
├── NOTEBOOK_SPECIFICATIONS.md    # This full spec (100+ pages)
├── SPECIFICATION_SUMMARY.md      # This executive summary
├── IMPLEMENTATION_GUIDE.md       # Step-by-step developer guide
├── TROUBLESHOOTING.md            # Common issues and solutions
└── API_REFERENCE.md              # Function signatures and usage
```

---

## Questions & Answers

### Q1: Why 20-30 hours for Notebook 02?
**A**: Processing 11,979 samples at ~40 seconds each = 132 hours. With optimizations (parallel downloads, batch processing), we target 20-30 hours. Use Colab background execution.

### Q2: Can we use only 1000 samples for faster testing?
**A**: Yes for initial testing, but the specification requires processing the full 11,979 samples for the final deliverable. The checkpoint system makes this feasible.

### Q3: What if we can't get GPU access?
**A**: Notebook 03 will still work on CPU, just slower (30 min vs 10 min). Notebook 04 doesn't require GPU. Notebook 02 doesn't use GPU at all.

### Q4: How do we handle failed samples?
**A**: Each notebook logs failures and continues processing. Final CSV includes a `success_flag` column. Target is >90% success, not 100%.

### Q5: Can these run locally instead of Colab?
**A**: Yes, but you'll need to manage dependencies manually. Colab is recommended for consistency and GPU access.

---

## Resources

### Full Documentation
- **Complete Spec**: [`NOTEBOOK_SPECIFICATIONS.md`](./NOTEBOOK_SPECIFICATIONS.md) (detailed technical spec)
- **Project Memory**: [`../PROJECT_MEMORY.md`](../PROJECT_MEMORY.md) (historical context)
- **Development Guide**: [`../CLAUDE.md`](../CLAUDE.md) (Claude Code instructions)

### External Resources
- [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)
- [Lightkurve Tutorials](https://docs.lightkurve.org/tutorials/)
- [XGBoost GPU Training](https://xgboost.readthedocs.io/en/stable/gpu/)
- [Google Colab Guide](https://colab.research.google.com/)

### Support
- GitHub Issues: Track progress and blockers
- Project Memory: Historical decisions and solutions
- SPARC Methodology: Specification-first development

---

**Status**: ✅ READY FOR IMPLEMENTATION
**Next Milestone**: Notebook 02 Feature Extraction
**Timeline**: Start today, complete in 3-4 days (including execution)

---

*Generated using SPARC methodology*
*Last updated: 2025-09-30*
*Version: 1.0*