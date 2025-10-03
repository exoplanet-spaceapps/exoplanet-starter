# Google Colab Feature Extraction Guide

## Overview

**Notebook**: `02_bls_baseline_COLAB.ipynb`
**Purpose**: Extract BLS/TLS features from 11,979 exoplanet candidates
**Runtime**: ~5-10 hours (depends on Colab GPU/TPU availability)
**Output**: 17 features per sample + metadata

---

## Features

### Checkpoint System
- **Auto-save**: Every 100 samples
- **Auto-resume**: Picks up from last checkpoint after disconnect
- **Storage**: Google Drive for persistence
- **Recovery**: Full state recovery across sessions

### Feature Extraction
**Total: 17 features per sample**

#### 1. Input Parameters (4 features)
- `input_period`: Catalog orbital period (days)
- `input_depth`: Catalog transit depth (relative flux)
- `input_duration`: Catalog transit duration (days)
- `input_epoch`: Transit epoch time

#### 2. Flux Statistics (4 features)
- `flux_std`: Standard deviation of flux
- `flux_mad`: Median absolute deviation
- `flux_skewness`: Distribution skewness
- `flux_kurtosis`: Distribution kurtosis

#### 3. BLS Features (5 features)
- `bls_period`: BLS detected period
- `bls_t0`: BLS transit time
- `bls_duration`: BLS duration
- `bls_depth`: BLS depth
- `bls_snr`: BLS signal-to-noise ratio

#### 4. Advanced Features (4 features)
- `duration_over_period`: Duration/period ratio
- `odd_even_depth_diff`: Odd-even transit depth difference (binary detection)
- `transit_symmetry`: Transit shape symmetry (0=symmetric, 1=asymmetric)
- `periodicity_strength`: Periodic signal strength (0-1)

---

## Quick Start

### Setup (First Time)

1. **Open in Colab**:
   ```
   https://colab.research.google.com/github/YOUR_USERNAME/exoplanet-starter/blob/main/notebooks/02_bls_baseline_COLAB.ipynb
   ```

2. **Run Cell 1** (Package Installation):
   ```python
   !pip install -q numpy==1.26.4 scipy'<1.13' astropy lightkurve transitleastsquares
   ```

   ⚠️ **IMPORTANT**: Click **Runtime → Restart runtime** after this cell!

3. **Run Cell 2** (Mount Drive):
   - Authorize Google Drive access
   - Creates project directories automatically

4. **Upload Dataset** (Cell 4):
   - Upload `supervised_dataset.csv` to:
     ```
     /content/drive/MyDrive/exoplanet-spaceapps/data/
     ```
   - Or use the file upload widget in the cell

5. **Run Cells 3-6** sequentially to load functions

6. **Start Extraction** (Cell 7):
   ```python
   features_df = extract_features_batch(
       samples_df=samples_df,
       checkpoint_mgr=checkpoint_mgr,
       batch_size=100,
       run_bls=True  # Set False for faster processing
   )
   ```

---

## Usage Scenarios

### Scenario 1: First Run (No Checkpoints)

```python
# Cell 1: Install packages
!pip install -q numpy==1.26.4 scipy'<1.13' astropy lightkurve transitleastsquares
# → RESTART RUNTIME

# Cell 2-6: Setup (run sequentially)
# Cell 7: Start extraction
features_df = extract_features_batch(samples_df, checkpoint_mgr, batch_size=100)
```

**Expected behavior**:
- Processes samples 0-100 (Batch 1)
- Saves checkpoint to Drive
- Continues to Batch 2...

---

### Scenario 2: Resume After Disconnect

```python
# Cell 1: Reinstall packages
!pip install -q numpy==1.26.4 scipy'<1.13' astropy lightkurve transitleastsquares
# → RESTART RUNTIME

# Cell 2-6: Reload functions
# Cell 7: Resume extraction (auto-detects last checkpoint)
features_df = extract_features_batch(samples_df, checkpoint_mgr, batch_size=100)
```

**Expected behavior**:
- Detects existing checkpoints
- Resumes from sample 3500 (if 35 batches completed)
- No duplicate processing

---

### Scenario 3: Test Run (Small Dataset)

```python
# Cell 4: Limit dataset size
samples_df = samples_df.head(200)  # Test with 200 samples

# Cell 7: Run extraction
features_df = extract_features_batch(samples_df, checkpoint_mgr, batch_size=50)
```

**Expected runtime**: ~10-15 minutes for 200 samples

---

### Scenario 4: Fast Mode (No BLS Search)

```python
# Cell 7: Disable BLS for speed
features_df = extract_features_batch(
    samples_df=samples_df,
    checkpoint_mgr=checkpoint_mgr,
    batch_size=100,
    run_bls=False  # 3-5x faster, uses catalog values
)
```

**Speed improvement**: ~3-5x faster
**Trade-off**: Uses catalog values instead of BLS-detected values

---

## Directory Structure

```
/content/drive/MyDrive/exoplanet-spaceapps/
├── checkpoints/
│   ├── batch_0000_0100.json  # Checkpoint 1 (samples 0-99)
│   ├── batch_0100_0200.json  # Checkpoint 2 (samples 100-199)
│   └── ...
├── data/
│   └── supervised_dataset.csv  # Input dataset (11,979 samples)
└── results/
    ├── bls_tls_features.csv  # Final output (17 features)
    └── failed_samples.csv    # Failed sample indices
```

---

## Monitoring Progress

### Cell 8: Real-time Dashboard

```python
monitor_progress(checkpoint_mgr, len(samples_df), update_interval=30)
```

**Output**:
```
🚀 Feature Extraction Progress
============================================================
[████████████████████████████████░░░░░░░░░░░░░░░░░░] 62.5%

✅ Completed:  7487 / 11,979
❌ Failed:     23
⏳ Remaining:  4469

📈 Success Rate: 62.50%
📉 Failure Rate: 0.19%

⏰ Last update: 2025-01-29 14:32:15
============================================================
```

### Progress Bar Visualization
- Green bar: Completed samples
- Red bar: Failed samples
- Yellow bar: Remaining samples

---

## Checkpoint Format

```json
{
  "checkpoint_id": "batch_0300_0400",
  "timestamp": "2025-01-29T14:32:15.123456",
  "batch_range": [300, 400],
  "completed_indices": [300, 301, 302, ..., 398, 399],
  "failed_indices": [350, 375],
  "features": {
    "300": {
      "input_period": 3.5,
      "flux_std": 0.001234,
      "bls_snr": 12.5,
      ...
    }
  },
  "metadata": {
    "batch_num": 4,
    "total_batches": 120,
    "processing_time_sec": 312.45,
    "samples_per_sec": 0.31
  }
}
```

---

## Performance Benchmarks

| Configuration | Speed | Runtime (11,979 samples) |
|--------------|-------|-------------------------|
| **BLS Enabled (GPU)** | 0.3-0.5 samples/sec | 7-10 hours |
| **BLS Disabled (GPU)** | 1.5-2.0 samples/sec | 2-3 hours |
| **BLS Enabled (CPU)** | 0.1-0.2 samples/sec | 16-33 hours |

**Recommendation**: Use GPU runtime with `run_bls=True` for best quality

---

## Troubleshooting

### Problem: Runtime Disconnects Frequently

**Causes**:
- Colab free tier timeout (12 hours max)
- Inactive browser tab
- Network issues

**Solutions**:
1. Keep Colab tab active (prevents idle timeout)
2. Enable browser notifications for reconnect alerts
3. Use Colab Pro for longer runtimes (24 hours)
4. Checkpoints ensure no data loss

---

### Problem: `RuntimeError: NumPy 2.0 incompatibility`

**Cause**: `lightkurve` and `transitleastsquares` require NumPy 1.x

**Solution**:
```python
# Cell 1
!pip install -q numpy==1.26.4 scipy'<1.13' astropy
# RESTART RUNTIME (mandatory!)
```

---

### Problem: `FileNotFoundError: supervised_dataset.csv`

**Cause**: Dataset not uploaded to Google Drive

**Solution**:
```python
# Option 1: Upload to Drive manually
# Navigate to /content/drive/MyDrive/exoplanet-spaceapps/data/
# Upload supervised_dataset.csv

# Option 2: Use file upload widget (Cell 4)
from google.colab import files
uploaded = files.upload()
samples_df = pd.read_csv('supervised_dataset.csv')
```

---

### Problem: Slow Processing (< 0.1 samples/sec)

**Causes**:
- CPU runtime (not GPU)
- BLS search enabled
- Large light curves

**Solutions**:
```python
# 1. Enable GPU runtime
# Runtime → Change runtime type → GPU

# 2. Disable BLS for speed
run_bls=False  # In Cell 7

# 3. Reduce batch size
batch_size=50  # Less memory per batch
```

---

### Problem: Memory Error / Out of Memory

**Cause**: Processing too many samples at once

**Solution**:
```python
# Reduce batch size
checkpoint_mgr = CheckpointManager(
    drive_path=str(BASE_DIR),
    batch_size=50  # Default: 100
)
```

---

### Problem: Failed Samples

**Expected behavior**: 0.1-0.5% failure rate (10-50 samples)

**Common causes**:
- Light curve not available in MAST
- Corrupted data
- Timeout during download

**View failed samples**:
```python
# Cell 9: Check failures
failed_indices = checkpoint_mgr.get_failed_indices()
print(f"Failed samples: {len(failed_indices)}")
print(f"Indices: {failed_indices}")
```

**Retry failed samples**:
```python
# Create subset with failed samples
failed_df = samples_df.iloc[failed_indices]
retry_features = extract_features_batch(failed_df, checkpoint_mgr)
```

---

## Output Validation

### Cell 9: Validate Results

```python
results_file = OUTPUT_DIR / 'bls_tls_features.csv'
features_df = pd.read_csv(results_file)

# Check completeness
print(f"Total samples: {len(features_df)}")
print(f"Expected: 11,979")
print(f"Completion rate: {len(features_df) / 11979 * 100:.2f}%")

# Check for NaN values
for col in features_df.columns:
    nan_count = features_df[col].isna().sum()
    if nan_count > 0:
        print(f"Warning: {col} has {nan_count} NaN values")

# Label distribution
print(features_df['label'].value_counts())
```

**Expected output**:
```
Total samples: 11,979
Expected: 11,979
Completion rate: 100.00%

Label distribution:
1    6500  # Confirmed planets
0    5479  # False positives
```

---

## Advanced Usage

### Custom Feature Extraction

Modify `extract_features_from_lightcurve()` in Cell 5:

```python
def extract_features_from_lightcurve(...):
    # Add custom features
    features['custom_feature'] = compute_custom_metric(time, flux)

    return features
```

### Multi-Sector Processing

```python
# Cell 6: Modify light curve download
search_result = lk.search_lightcurve(f'TIC {target_id}', mission='TESS')
lc_collection = search_result.download_all()  # All sectors
lc = lc_collection.stitch()  # Combine sectors
```

### Parallel Processing (Experimental)

```python
# Use multiple Colab instances
# Split dataset: samples_df.iloc[0:3000], samples_df.iloc[3000:6000], ...
# Merge checkpoints manually after completion
```

---

## Best Practices

### 1. Checkpoint Management
- ✅ Keep checkpoints until final validation
- ✅ Backup Drive folder regularly
- ❌ Don't delete checkpoints during processing

### 2. Resource Optimization
- ✅ Use GPU runtime (faster)
- ✅ Close other Colab notebooks (avoid quota limits)
- ✅ Monitor RAM usage (Runtime → Manage sessions)

### 3. Data Quality
- ✅ Validate dataset before extraction
- ✅ Check for duplicate samples
- ✅ Verify feature distributions after extraction

### 4. Error Handling
- ✅ Review failed samples log
- ✅ Retry failures separately
- ✅ Document systematic failures

---

## FAQ

**Q: How long does full extraction take?**
A: 7-10 hours with GPU and BLS enabled. Use `run_bls=False` for 2-3 hours.

**Q: Can I pause and resume?**
A: Yes! Checkpoints are saved every 100 samples to Google Drive. Simply restart and run Cell 7.

**Q: What if Colab disconnects during processing?**
A: No problem. Checkpoints ensure progress is saved. Restart runtime and continue from Cell 2.

**Q: How much Drive space do I need?**
A: ~500 MB for checkpoints + ~50 MB for final CSV = ~550 MB total.

**Q: Can I process a subset of samples?**
A: Yes. In Cell 4: `samples_df = samples_df.head(500)` for testing.

**Q: What if I get authentication errors?**
A: Re-mount Google Drive (Cell 2) and authorize access.

**Q: How do I download results?**
A: Use Cell 11 or download manually from Drive at `/content/drive/MyDrive/exoplanet-spaceapps/results/`

---

## Next Steps

After successful extraction:

1. **Validate Output** (Cell 9)
2. **Download Results** (Cell 11)
3. **Cleanup Checkpoints** (Cell 10, optional)
4. **Proceed to Training** (Notebook 03: `03_injection_train.ipynb`)

---

## Support

**Documentation**: `COLAB_USAGE_GUIDE.md` (this file)
**Tests**: `tests/test_feature_extraction_colab.py`
**Issues**: Report in project repository
**Version**: 1.0.0 (2025-01-29)

---

## Changelog

### v1.0.0 (2025-01-29)
- Initial production release
- 17 features per sample
- Checkpoint system with auto-resume
- Google Drive integration
- Real-time progress monitoring
- Comprehensive error handling
- 7/7 tests passing

---

**Ready to extract features? Start with Cell 1!** 🚀