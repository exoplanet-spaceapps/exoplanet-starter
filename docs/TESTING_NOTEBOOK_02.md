# 📋 Notebook 02 Testing Guide

## 🎯 Overview

A comprehensive testing suite has been added to **02_bls_baseline.ipynb** (Cell 8) to validate all critical components before running the full feature extraction pipeline.

## 🧪 Test Suite Components

### Test 1: NumPy Version Verification
**Purpose**: Ensure NumPy 1.26.x is installed (required for transitleastsquares compatibility)

**Pass Criteria**:
- NumPy version starts with "1.26"
- Example: `1.26.4` ✅

**Failure Action**:
```bash
pip install numpy==1.26.4
```

---

### Test 2: Checkpoint System Functionality
**Purpose**: Validate checkpoint save/resume mechanism for batch processing

**Pass Criteria**:
- Successfully creates checkpoint directory
- Saves checkpoint data to JSON
- Resumes from last checkpoint correctly
- Returns correct batch number (1 after saving batch 0)

**Test Scope**:
- Temporary directory creation
- JSON serialization
- File I/O operations
- Cleanup after test

---

### Test 3: Single Sample Feature Extraction
**Purpose**: Verify end-to-end feature extraction pipeline with known target

**Test Target**: TIC 25155310 (TOI-270, known multi-planet system)

**Pass Criteria**:
- Downloads light curve from MAST successfully
- Runs BLS algorithm without errors
- Extracts 8+ features:
  - `tic_id`
  - `bls_period` (1.0-15.0 days)
  - `bls_power`
  - `bls_depth`
  - `bls_duration`
  - `num_points`
  - `flux_std`
  - `flux_median`
- No NaN values in features
- Valid period range (1.0 <= period <= 15.0)

**Note**: This test may skip if MAST is unavailable (normal for offline testing)

---

### Test 4: Google Drive Access (Colab Only)
**Purpose**: Ensure Google Drive is mounted and writable

**Pass Criteria** (Colab):
- `/content/drive/MyDrive/spaceapps-exoplanet/checkpoints/` is writable
- Can create and delete test files

**Behavior** (Local):
- Skips test automatically
- Uses `./checkpoints/` for local execution

**Failure Action** (Colab):
1. Remount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive', force_remount=True)
   ```
2. Check permissions on Google Drive folder
3. Verify sufficient storage space

---

### Test 5: Batch Processing (Small Scale)
**Purpose**: Test batch processing pipeline with 5 samples

**Data Source**: `data/supervised_dataset.csv`

**Pass Criteria**:
- Loads CSV successfully
- Processes 5 sample TIC IDs
- Success rate >= 40% (2/5 samples)

**Note**: Low success rate may indicate MAST server availability issues (temporary, retry later)

---

## 📊 Test Output Format

### Success Example:
```
============================================================
🧪 Running Notebook 02 Validation Tests...
============================================================

Test 1/5: NumPy version compatibility...
  ✅ NumPy 1.26.4 detected (compatible)

Test 2/5: Checkpoint system functionality...
  ✅ Checkpoint system working (resumed batch: 1)

Test 3/5: Feature extraction (single target)...
  📡 Testing with TIC 25155310 (TOI-270)...
  ✅ Extracted 8 features successfully
     - Period: 3.360 days
     - Power: 0.8542
     - Data points: 18362

Test 4/5: Google Drive access...
  ✅ Google Drive writable at /content/drive/MyDrive/spaceapps-exoplanet/checkpoints

Test 5/5: Batch processing (small scale)...
  📊 Testing with 5 samples...
  ✅ Batch test: 60.0% success rate (3/5)

============================================================
📊 TEST SUMMARY
============================================================
✅ PASS     - NumPy version
✅ PASS     - Checkpoint system
✅ PASS     - Feature extraction
✅ PASS     - Google Drive access
✅ PASS     - Batch processing
------------------------------------------------------------
Results: 5 passed, 0 failed, 0 skipped
============================================================
✅ All critical tests passed! Ready for production run.
   You can now proceed with full feature extraction.
============================================================
```

### Failure Example:
```
============================================================
📊 TEST SUMMARY
============================================================
❌ FAIL     - NumPy version
✅ PASS     - Checkpoint system
⚠️  SKIP     - Feature extraction
✅ PASS     - Google Drive access
⚠️  SKIP     - Batch processing
------------------------------------------------------------
Results: 2 passed, 1 failed, 2 skipped
============================================================
⚠️  Some tests failed. Please review errors above.
   Fix issues before running full extraction.
============================================================
```

---

## 🚀 How to Run Tests

### In Colab:
1. Open `notebooks/02_bls_baseline.ipynb` in Colab
2. Run Cells 1-7 (dependencies and configuration)
3. **Run Cell 8 (TEST SUITE)** ⭐
4. Review test results
5. If all pass, proceed with Cell 9+

### Locally:
1. Ensure NumPy 1.26.4 is installed
2. Have `data/supervised_dataset.csv` available
3. Run test cell
4. Tests 3, 4, 5 may skip (normal for local testing)

---

## ✅ Validation Checklist

Before running full feature extraction (`max_samples=2000`):

- [ ] NumPy 1.26.x installed
- [ ] Checkpoint system works
- [ ] Feature extraction succeeds for test target
- [ ] Google Drive accessible (Colab) or local directory exists
- [ ] Batch processing has >= 40% success rate
- [ ] All critical tests passed (minimum 3/5)

---

## 🛠️ Troubleshooting

### Test 1 Fails (NumPy)
```bash
# In Colab (Cell 4)
!pip install -q numpy==1.26.4 scipy'<1.13' astropy
# Restart Runtime → Continue from Cell 6
```

### Test 3 Skips (MAST Unavailable)
- **Cause**: MAST archive server temporarily down
- **Action**: Wait 10-30 minutes and retry
- **Alternative**: Use cached data if available

### Test 4 Fails (Drive Access)
```python
# Remount Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

### Test 5 Low Success Rate
- **Cause**: MAST server rate limiting or downtime
- **Action**:
  1. Wait 30 minutes
  2. Reduce `max_samples` to 500-1000
  3. Enable longer delays between requests

---

## 📝 Implementation Details

### Script Location
`C:\Users\thc1006\Desktop\dev\exoplanet-starter\scripts\insert_test_cell_safe.py`

### Cell Position
- **Notebook**: `notebooks/02_bls_baseline.ipynb`
- **Cell Number**: 8 (inserted after configuration, before execution)
- **Cell Type**: Code

### Dependencies Used
- `numpy`: Version checking
- `pandas`: Data handling
- `lightkurve`: Light curve download
- `astropy.timeseries.BoxLeastSquares`: BLS algorithm
- `tempfile`: Temporary directories
- `json`: Checkpoint serialization
- `pathlib`: Path handling

---

## 🔄 Continuous Integration

This test suite can be adapted for CI/CD:

```python
# Run tests and exit with status code
if __name__ == '__main__':
    passed = sum(1 for _, result in test_results if result is True)
    failed = sum(1 for _, result in test_results if result is False)

    if failed > 0:
        sys.exit(1)  # CI failure
    elif passed < 3:
        sys.exit(2)  # Insufficient coverage
    else:
        sys.exit(0)  # Success
```

---

## 📚 References

- **Project Memory**: `PROJECT_MEMORY.md`
- **Main README**: `README.md`
- **Development Guide**: `CLAUDE.md`
- **Original Notebook**: `notebooks/02_bls_baseline.ipynb`

---

**Last Updated**: 2025-01-29
**Version**: 1.0.0
**Status**: ✅ Production Ready