# Notebook 02 BLS Baseline - Local Execution Guide

## Quick Start

### Current Status
✅ **Conversion Complete**: Notebook 02 has been successfully converted from Google Colab to local CPU execution
🔄 **Test Run Active**: Processing 100-sample subset (currently at sample 6/100)
⏱️ **Estimated Test Completion**: ~60 minutes from start

## What Was Done

### 1. Package Installation ✓
All required Python packages installed:
- Scientific: numpy, pandas, scipy, matplotlib, scikit-learn
- Astronomy: astropy (7.1.0), lightkurve (2.5.1), astroquery (0.4.11)
- Transit Analysis: transitleastsquares (1.32), wotan (1.10)
- Utilities: tqdm, numba

### 2. Code Conversion ✓
- Removed all Google Colab dependencies (`google.colab` imports)
- Updated file paths from `/content/drive/` to local Windows paths
- Fixed import statements for new astropy API (`BoxLeastSquares` moved to `timeseries`)
- Removed Unicode characters causing encoding issues on Windows console

### 3. Optimizations Added ✓
- **Checkpointing**: Saves progress every 10 samples
- **Resume Capability**: Can restart from last checkpoint
- **Error Handling**: Robust error catching and failure logging
- **Progress Tracking**: Real-time tqdm progress bars
- **Configurable Limits**: Can process subset for testing

## Files Created

### Scripts (`/scripts`)
1. **`run_bls_analysis_optimized.py`** ⭐ RECOMMENDED
   - Production-ready with checkpointing
   - Currently configured for 100-sample test
   - ~60 minutes for 100 samples
   - ~4-5 days for full 11,979 samples

2. **`run_bls_analysis.py`**
   - Basic version without checkpointing
   - Use for quick tests only

3. **`check_progress.py`**
   - Monitor processing progress
   - View statistics and failure reasons

4. **`generate_final_report.py`**
   - Generate comprehensive execution report
   - Statistics and recommendations

### Documentation (`/docs`)
- **`BLS_ANALYSIS_REPORT.md`** - Complete technical report

## Usage

### Monitor Current Test Run
```bash
cd "C:\Users\tingy\Desktop\dev\exoplanet-starter"

# Check progress
python scripts/check_progress.py

# View full report
python scripts/generate_final_report.py
```

### Run Full Analysis (All 11,979 Samples)
```bash
# 1. Edit the configuration
# Open: scripts/run_bls_analysis_optimized.py
# Change: MAX_SAMPLES = None  (line 29)

# 2. Run the script
python scripts/run_bls_analysis_optimized.py

# 3. Monitor progress (in another terminal)
python scripts/check_progress.py
```

### Resume After Interruption
The script automatically resumes from the last checkpoint:
```bash
# Just run it again - it will resume automatically
python scripts/run_bls_analysis_optimized.py
```

## Processing Details

### What the Script Does
1. **Downloads Light Curves**: Fetches TESS light curves from MAST archive
2. **Preprocessing**: Removes NaNs, filters outliers (5-sigma clipping)
3. **BLS Analysis**: Box Least Squares period search
   - Period range: 0.5-20 days
   - Duration range: 0.05-0.3 days
4. **Feature Extraction**: Extracts period, power, duration, depth, SNR
5. **Saves Results**: CSV files with all extracted features

### Performance
- **Single Sample**: ~10-30 seconds (varies by data size and network speed)
- **100 Samples**: ~60 minutes
- **11,979 Samples**: ~4-5 days (continuous running)

### Output Files
- **`data/bls_results.csv`** - Successfully processed samples with BLS features
- **`data/bls_failed_samples.csv`** - Failed samples with error reasons
- **`data/bls_results_checkpoint_N.csv`** - Checkpoint files (every 10 samples)

## System Requirements

### Minimum
- **Python**: 3.10+ (tested on 3.13.3)
- **RAM**: 4 GB
- **Disk Space**: 10 GB free (for data cache)
- **Internet**: Stable connection required (downloads from MAST)

### Recommended
- **Python**: 3.13+
- **RAM**: 8 GB+
- **Disk Space**: 20 GB+ free
- **Internet**: 10+ Mbps

## Troubleshooting

### Issue: Slow Processing
**Symptom**: Taking >60s per sample
**Cause**: Slow network or MAST server congestion
**Solution**:
- Check internet connection
- Try running during off-peak hours
- Consider cloud execution (AWS, GCP, Azure)

### Issue: Script Crashes
**Symptom**: Script stops unexpectedly
**Cause**: Memory issues or network interruption
**Solution**:
- Resume using the same command (uses checkpoints)
- Check available RAM
- Close other applications

### Issue: Many Failed Samples
**Symptom**: High failure rate in `bls_failed_samples.csv`
**Cause**: Data not available on MAST or poor quality
**Solution**:
- This is expected - not all targets have good data
- Check failure reasons in the CSV file
- Failed samples are automatically logged

### Issue: Import Errors
**Symptom**: `ModuleNotFoundError` or `ImportError`
**Cause**: Missing packages
**Solution**:
```bash
# Reinstall packages
pip install numpy pandas astropy matplotlib scikit-learn scipy
pip install lightkurve astroquery transitleastsquares wotan tqdm
```

## Dataset Information

- **File**: `data/supervised_dataset.csv`
- **Total Samples**: 11,979
- **Columns**: label, source, toi, tid, target_id, period, depth, duration, kepid
- **Label Distribution**:
  - Label 0 (Non-planet): 6,035 (50.4%)
  - Label 1 (Planet): 5,944 (49.6%)

## Next Steps

### After Test Completion (100 samples)
1. Review results:
   ```bash
   python scripts/generate_final_report.py
   ```
2. Verify output quality
3. Decide: Run full analysis or use test data

### For Full Processing (11,979 samples)
1. Configure for full run (set `MAX_SAMPLES = None`)
2. Run script in background or screen session
3. Monitor progress periodically
4. Wait ~4-5 days for completion

### After Full Completion
1. Load `bls_results.csv` in Notebook 03
2. Use BLS features for exoplanet classification
3. Train machine learning models

## Performance Optimization Options

### Option 1: Parallel Processing (Future)
- Modify script to use `multiprocessing`
- Process 4-8 samples simultaneously
- Reduce time by 4-8x
- **Estimated Time**: ~12-30 hours for full dataset

### Option 2: Cloud Execution
- Use AWS EC2, Google Cloud, or Azure VM
- Leverage multiple cores
- Better network bandwidth to MAST
- **Estimated Time**: ~6-12 hours for full dataset

### Option 3: Pre-cached Data
- Use pre-downloaded TESS light curves
- Skip download step
- **Estimated Time**: ~4-8 hours for full dataset

## Background Execution

### Windows (PowerShell)
```powershell
# Start in background
Start-Process python -ArgumentList "scripts/run_bls_analysis_optimized.py" -NoNewWindow -RedirectStandardOutput "logs/output.log" -RedirectStandardError "logs/error.log"

# Monitor progress
Get-Content logs/output.log -Wait -Tail 50
```

### Linux/Mac (bash)
```bash
# Start in background
nohup python scripts/run_bls_analysis_optimized.py > logs/output.log 2>&1 &

# Monitor progress
tail -f logs/output.log
```

## Key Configuration Variables

In `run_bls_analysis_optimized.py`:

```python
# Line 28-30
CHECKPOINT_INTERVAL = 10  # Save every N samples (10-100 recommended)
MAX_SAMPLES = 100         # Limit samples (None = all 11,979)
RESUME_FROM_CHECKPOINT = True  # Auto-resume capability
```

## Support & Documentation

### Files to Reference
- **Technical Details**: `docs/BLS_ANALYSIS_REPORT.md`
- **This Guide**: `README_BLS_EXECUTION.md`
- **Original Notebook**: `notebooks/02_bls_baseline.ipynb`

### Common Commands
```bash
# Check progress
python scripts/check_progress.py

# Generate report
python scripts/generate_final_report.py

# Start processing
python scripts/run_bls_analysis_optimized.py

# Resume processing
python scripts/run_bls_analysis_optimized.py  # Same command
```

## Success Criteria

### Test Run (100 samples)
- ✓ Environment setup complete
- ✓ All packages installed
- ✓ Script running successfully
- 🔄 Processing 6/100 samples (currently active)
- ⏳ Estimated completion: ~50 minutes remaining

### Full Run (11,979 samples)
- ⏳ Not started yet
- Estimated time: 4-5 days
- Requires configuration change: `MAX_SAMPLES = None`

---

**Status**: ✅ Conversion Complete | 🔄 Test Run In Progress (6/100)
**Last Updated**: 2025-09-30
**Python Version**: 3.13.3
**Operating System**: Windows 10/11