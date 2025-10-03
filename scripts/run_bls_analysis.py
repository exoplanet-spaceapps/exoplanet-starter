#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Execute BLS Baseline Analysis (Notebook 02) - Local CPU Version
Processes all 11979 samples from supervised_dataset.csv
"""
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Set up environment
os.chdir(r'C:\Users\tingy\Desktop\dev\exoplanet-starter')
sys.path.insert(0, os.getcwd())

print("="*80)
print("BLS Baseline Analysis - Local CPU Execution")
print("="*80)
print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print()

# ============================================================================
# CELL 1: Environment Setup (Skip Colab-specific parts)
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

# TLS and wotan for advanced analysis
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

# For progress tracking
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
print()

# ============================================================================
# CELL 4: Helper Functions
# ============================================================================
print("[Step 4] Defining helper functions...")

def download_lightcurve(target_id, mission='TESS', exptime=120):
    """Download light curve from MAST."""
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
        print(f"Error downloading {target_id}: {e}")
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

        # Extract best period
        best_period = periodogram.period[np.argmax(periodogram.power)]
        best_power = np.max(periodogram.power)
        best_duration = periodogram.duration[np.argmax(periodogram.power)]
        best_t0 = periodogram.transit_time[np.argmax(periodogram.power)]

        # Get statistics
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
        print(f"BLS error: {e}")
        return None

print("[OK] Helper functions defined\n")

# ============================================================================
# CELL 5: Main Processing Loop
# ============================================================================
print("[Step 5] Starting BLS analysis on all samples...")
print(f"Total samples to process: {len(df)}")
print()

# Results storage
results = []
failed_samples = []

# Process each sample
for idx in tqdm(range(len(df)), desc="Processing samples"):
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
        failed_samples.append({'index': idx, 'target_id': target_id, 'reason': str(e)})
        continue

print()
print("="*80)
print("PROCESSING COMPLETE")
print("="*80)
print(f"Successfully processed: {len(results)}")
print(f"Failed: {len(failed_samples)}")
print()

# ============================================================================
# CELL 6: Save Results
# ============================================================================
print("[Step 6] Saving results...")

# Save successful results
results_df = pd.DataFrame(results)
output_path = r'C:\Users\tingy\Desktop\dev\exoplanet-starter\data\bls_results.csv'
results_df.to_csv(output_path, index=False)
print(f"[OK] Results saved to: {output_path}")

# Save failed samples
if failed_samples:
    failed_df = pd.DataFrame(failed_samples)
    failed_path = r'C:\Users\tingy\Desktop\dev\exoplanet-starter\data\bls_failed_samples.csv'
    failed_df.to_csv(failed_path, index=False)
    print(f"[OK] Failed samples saved to: {failed_path}")

# ============================================================================
# CELL 7: Summary Statistics
# ============================================================================
print()
print("[Step 7] Summary statistics:")
print(f"Total processed: {len(results)}/{len(df)} ({len(results)/len(df)*100:.1f}%)")
print()

if len(results) > 0:
    print("BLS Results Summary:")
    print(results_df.describe())
    print()
    print("Results by label:")
    print(results_df.groupby('label').size())

print()
print("="*80)
print("ANALYSIS COMPLETE!")
print("="*80)