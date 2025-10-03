#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Execute BLS Baseline Analysis (Notebook 02) - Optimized Local CPU Version
- Processes samples with checkpointing to avoid data loss
- Saves intermediate results every N samples
- Can resume from last checkpoint
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Set up environment
os.chdir(r'C:\Users\tingy\Desktop\dev\exoplanet-starter')
sys.path.insert(0, os.getcwd())

print("="*80)
print("BLS Baseline Analysis - Optimized Local CPU Execution")
print("="*80)
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print()

# ============================================================================
# Configuration
# ============================================================================
CHECKPOINT_INTERVAL = 10  # Save results every N samples
MAX_SAMPLES = None  # Set to a number for testing (e.g., 100), None for all
RESUME_FROM_CHECKPOINT = True  # Resume from last checkpoint if available

# ============================================================================
# CELL 1: Environment Setup
# ============================================================================
print("[Step 1] Environment setup...")
IN_COLAB = False
print("[OK] Local environment - packages already installed\n")

# ============================================================================
# CELL 2: Import Libraries
# ============================================================================
print("[Step 2] Importing libraries...")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from astropy.timeseries import TimeSeries, BoxLeastSquares
from astropy.time import Time
import lightkurve as lk
from scipy import stats
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# TLS and wotan
try:
    from transitleastsquares import transitleastsquares as tls
    print("[OK] TransitLeastSquares imported")
except ImportError:
    print("[WARNING] TransitLeastSquares not available")

try:
    from wotan import flatten
    print("[OK] Wotan imported")
except ImportError:
    print("[WARNING] Wotan not available")

from tqdm.auto import tqdm
print("[OK] All core libraries imported\n")

# ============================================================================
# CELL 3: Load Dataset
# ============================================================================
print("[Step 3] Loading dataset...")
data_path = r'C:\Users\tingy\Desktop\dev\exoplanet-starter\data\supervised_dataset.csv'
df = pd.read_csv(data_path)
print(f"[OK] Loaded {len(df)} samples from supervised_dataset.csv")
print(f"  Columns: {list(df.columns)}")
print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

# Limit samples if specified
if MAX_SAMPLES:
    df = df.head(MAX_SAMPLES)
    print(f"[INFO] Limited to first {MAX_SAMPLES} samples for testing")
print()

# ============================================================================
# CELL 4: Helper Functions
# ============================================================================
print("[Step 4] Defining helper functions...")

def download_lightcurve(target_id, mission='TESS', exptime=120):
    """Download light curve from MAST with error handling."""
    try:
        search = lk.search_lightcurve(target_id, mission=mission, exptime=exptime)
        if len(search) == 0:
            return None
        lc = search.download_all()
        if lc is None:
            return None
        lc = lc.stitch()
        return lc
    except Exception as e:
        return None

def preprocess_lightcurve(lc):
    """Basic preprocessing: remove NaNs and outliers."""
    lc = lc.remove_nans()
    lc = lc.remove_outliers(sigma=5)
    return lc

def run_bls(time, flux, period_min=0.5, period_max=20, duration_min=0.05, duration_max=0.3):
    """Run Box Least Squares period search."""
    try:
        model = BoxLeastSquares(time * u.day, flux)
        periodogram = model.autopower(
            minimum_period=period_min,
            maximum_period=period_max,
            minimum_duration=duration_min,
            maximum_duration=duration_max
        )

        best_period = periodogram.period[np.argmax(periodogram.power)]
        best_power = np.max(periodogram.power)
        best_duration = periodogram.duration[np.argmax(periodogram.power)]
        best_t0 = periodogram.transit_time[np.argmax(periodogram.power)]

        stats_result = model.compute_stats(best_period, best_duration, best_t0)

        return {
            'period': best_period.value,
            'power': best_power,
            'duration': best_duration.value,
            't0': best_t0.value,
            'depth': stats_result['depth'],
            'snr': stats_result['snr'] if 'snr' in stats_result else 0
        }
    except Exception as e:
        return None

def save_checkpoint(results, failed_samples, checkpoint_num):
    """Save intermediate results."""
    results_df = pd.DataFrame(results)
    checkpoint_path = f'data/bls_results_checkpoint_{checkpoint_num}.csv'
    results_df.to_csv(checkpoint_path, index=False)

    if failed_samples:
        failed_df = pd.DataFrame(failed_samples)
        failed_path = f'data/bls_failed_checkpoint_{checkpoint_num}.csv'
        failed_df.to_csv(failed_path, index=False)

def load_last_checkpoint():
    """Load last checkpoint if available."""
    import glob
    checkpoints = glob.glob('data/bls_results_checkpoint_*.csv')
    if not checkpoints:
        return None, None, 0

    # Find latest checkpoint
    checkpoint_nums = [int(f.split('_')[-1].split('.')[0]) for f in checkpoints]
    latest_num = max(checkpoint_nums)

    results_df = pd.read_csv(f'data/bls_results_checkpoint_{latest_num}.csv')
    results = results_df.to_dict('records')

    failed_path = f'data/bls_failed_checkpoint_{latest_num}.csv'
    failed_samples = []
    if os.path.exists(failed_path):
        failed_df = pd.read_csv(failed_path)
        failed_samples = failed_df.to_dict('records')

    # Find last processed index
    processed_indices = set(r['index'] for r in results)
    processed_indices.update(f['index'] for f in failed_samples)
    last_index = max(processed_indices) if processed_indices else -1

    return results, failed_samples, last_index + 1

print("[OK] Helper functions defined\n")

# ============================================================================
# CELL 5: Load Checkpoint if Available
# ============================================================================
results = []
failed_samples = []
start_index = 0

if RESUME_FROM_CHECKPOINT:
    print("[Step 5a] Checking for previous checkpoint...")
    checkpoint_results, checkpoint_failed, start_index = load_last_checkpoint()
    if checkpoint_results is not None:
        results = checkpoint_results
        failed_samples = checkpoint_failed
        print(f"[OK] Resumed from checkpoint - already processed {start_index} samples")
        print(f"  Successful: {len(results)}, Failed: {len(failed_samples)}")
    else:
        print("[INFO] No checkpoint found, starting from beginning")
    print()

# ============================================================================
# CELL 6: Main Processing Loop
# ============================================================================
print("[Step 6] Starting BLS analysis...")
print(f"Total samples to process: {len(df)}")
print(f"Starting from index: {start_index}")
print(f"Checkpoint interval: every {CHECKPOINT_INTERVAL} samples")
print()

checkpoint_counter = 0

for idx in tqdm(range(start_index, len(df)), desc="Processing samples", initial=start_index, total=len(df)):
    row = df.iloc[idx]
    target_id = row['target_id']

    try:
        # Download light curve
        lc = download_lightcurve(target_id)
        if lc is None:
            failed_samples.append({'index': idx, 'target_id': target_id, 'reason': 'download_failed'})
            continue

        # Preprocess
        lc = preprocess_lightcurve(lc)
        if len(lc) < 100:
            failed_samples.append({'index': idx, 'target_id': target_id, 'reason': 'insufficient_data'})
            continue

        # Run BLS
        bls_result = run_bls(lc.time.value, lc.flux.value)
        if bls_result is None:
            failed_samples.append({'index': idx, 'target_id': target_id, 'reason': 'bls_failed'})
            continue

        # Store results
        result = {
            'index': idx,
            'label': row['label'],
            'target_id': target_id,
            'source': row.get('source', 'Unknown'),
            **bls_result
        }
        results.append(result)

    except Exception as e:
        failed_samples.append({'index': idx, 'target_id': target_id, 'reason': str(e)[:100]})
        continue

    # Save checkpoint
    checkpoint_counter += 1
    if checkpoint_counter >= CHECKPOINT_INTERVAL:
        save_checkpoint(results, failed_samples, idx)
        checkpoint_counter = 0

# Final checkpoint
if checkpoint_counter > 0:
    save_checkpoint(results, failed_samples, len(df))

print()
print("="*80)
print("PROCESSING COMPLETE")
print("="*80)
print(f"Successfully processed: {len(results)}")
print(f"Failed: {len(failed_samples)}")
print()

# ============================================================================
# CELL 7: Save Final Results
# ============================================================================
print("[Step 7] Saving final results...")

results_df = pd.DataFrame(results)
output_path = r'C:\Users\tingy\Desktop\dev\exoplanet-starter\data\bls_results.csv'
results_df.to_csv(output_path, index=False)
print(f"[OK] Results saved to: {output_path}")

if failed_samples:
    failed_df = pd.DataFrame(failed_samples)
    failed_path = r'C:\Users\tingy\Desktop\dev\exoplanet-starter\data\bls_failed_samples.csv'
    failed_df.to_csv(failed_path, index=False)
    print(f"[OK] Failed samples saved to: {failed_path}")

# ============================================================================
# CELL 8: Summary Statistics
# ============================================================================
print()
print("[Step 8] Summary statistics:")
print(f"Total processed: {len(results)}/{len(df)} ({len(results)/len(df)*100:.1f}%)")
print()

if len(results) > 0:
    print("BLS Results Summary:")
    print(results_df[['period', 'power', 'duration', 'depth', 'snr']].describe())
    print()
    print("Results by label:")
    print(results_df.groupby('label').size())

print()
print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)