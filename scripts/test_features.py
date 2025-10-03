#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test feature extraction from downloaded HDF5 files"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
warnings.filterwarnings('ignore')

print("="*70)
print("Exoplanet Detection - Test Feature Extraction")
print("="*70)

# Check dependencies
print("\n[1/7] Checking dependencies...")
try:
    from astropy.timeseries import BoxLeastSquares
    import scipy.stats
    print(f"  OK: astropy, scipy, numpy {np.__version__}, pandas {pd.__version__}")
except ImportError as e:
    print(f"  ERROR: {e}")
    print("  Install: pip install astropy scipy")
    sys.exit(1)

# Setup paths
DATA_DIR = PROJECT_ROOT / 'data'
LIGHTCURVE_DIR = DATA_DIR / 'lightcurves'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
OUTPUT_DIR = PROJECT_ROOT / 'data'

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n[2/7] Paths configured")
print(f"  Working: {PROJECT_ROOT}")
print(f"  Lightcurves: {LIGHTCURVE_DIR}")
print(f"  Output: {OUTPUT_DIR}")

# Config
CONFIG = {
    'bls_periods': 2000,
    'period_min': 0.5,
    'period_max': 15.0,
    'duration_min': 0.05,
    'duration_max': 0.3,  # FIXED: Must be < period_min
    'n_durations': 10,
}

print(f"\n[3/7] BLS Configuration")
for key, val in CONFIG.items():
    print(f"  {key}: {val}")

# Load dataset metadata
print(f"\n[4/7] Loading dataset metadata...")
dataset_path = DATA_DIR / 'supervised_dataset.csv'
if not dataset_path.exists():
    print(f"  ERROR: Dataset not found: {dataset_path}")
    sys.exit(1)

samples_df = pd.read_csv(dataset_path)
if 'sample_id' not in samples_df.columns:
    samples_df['sample_id'] = [f"SAMPLE_{i:06d}" for i in range(len(samples_df))]
if 'tic_id' not in samples_df.columns:
    if 'tid' in samples_df.columns:
        samples_df['tic_id'] = samples_df['tid']

print(f"  Total samples in dataset: {len(samples_df)}")

# Find downloaded files
h5_files = list(LIGHTCURVE_DIR.glob('*.h5'))
print(f"  Downloaded files: {len(h5_files)}")

# Extract sample IDs from filenames
downloaded_ids = set()
for f in h5_files:
    # Format: SAMPLE_000000_TIC12345.h5
    parts = f.stem.split('_')
    if len(parts) >= 2:
        sample_id = f"{parts[0]}_{parts[1]}"
        downloaded_ids.add(sample_id)

# Filter to only downloaded samples
available_samples = samples_df[samples_df['sample_id'].isin(downloaded_ids)].copy()
print(f"  Available for feature extraction: {len(available_samples)}")

if len(available_samples) == 0:
    print("  ERROR: No samples available for feature extraction")
    sys.exit(1)

# Feature extraction function
def extract_bls_features(h5_file_path, period_hint=None, duration_hint=None, depth_hint=None):
    """Extract BLS features from HDF5 file"""

    features = {
        'sample_id': None,
        'tic_id': None,
        'n_sectors': 0,
        'flux_mean': np.nan,
        'flux_std': np.nan,
        'flux_median': np.nan,
        'flux_mad': np.nan,
        'flux_skew': np.nan,
        'flux_kurt': np.nan,
        'bls_period': np.nan,
        'bls_duration': np.nan,
        'bls_depth': np.nan,
        'bls_power': np.nan,
        'bls_snr': np.nan,
        'period_match': np.nan,
        'duration_match': np.nan,
        'depth_match': np.nan,
        'status': 'failed',
        'error': None
    }

    try:
        with h5py.File(h5_file_path, 'r') as f:
            features['sample_id'] = f.attrs.get('sample_id', 'unknown')
            features['tic_id'] = f.attrs.get('tic_id', 0)
            features['n_sectors'] = f.attrs.get('n_sectors', 0)

            # Combine all sectors
            time_list = []
            flux_list = []

            for i in range(features['n_sectors']):
                sector = f[f'sector_{i}']
                time = np.array(sector['time'][:])
                flux = np.array(sector['flux'][:])

                # Filter out NaN values
                valid = ~(np.isnan(time) | np.isnan(flux))
                time_list.append(time[valid])
                flux_list.append(flux[valid])

            if len(time_list) == 0:
                features['error'] = 'no_valid_data'
                return features

            time = np.concatenate(time_list)
            flux = np.concatenate(flux_list)

            # Sort by time
            sort_idx = np.argsort(time)
            time = time[sort_idx]
            flux = flux[sort_idx]

            # Normalize flux
            flux_median = np.median(flux)
            flux = flux / flux_median

            # Basic statistics
            features['flux_mean'] = np.mean(flux)
            features['flux_std'] = np.std(flux)
            features['flux_median'] = np.median(flux)
            features['flux_mad'] = np.median(np.abs(flux - np.median(flux)))
            features['flux_skew'] = scipy.stats.skew(flux)
            features['flux_kurt'] = scipy.stats.kurtosis(flux)

            # BLS analysis
            periods = np.linspace(CONFIG['period_min'], CONFIG['period_max'], CONFIG['bls_periods'])
            durations = np.linspace(CONFIG['duration_min'], CONFIG['duration_max'], CONFIG['n_durations'])

            bls = BoxLeastSquares(time, flux)
            bls_result = bls.power(periods, durations)

            # Best period
            best_idx = np.argmax(bls_result.power)
            features['bls_period'] = bls_result.period[best_idx]
            features['bls_duration'] = bls_result.duration[best_idx]
            features['bls_depth'] = bls_result.depth[best_idx]
            features['bls_power'] = bls_result.power[best_idx]

            # Signal-to-noise ratio
            power_median = np.median(bls_result.power)
            power_std = np.std(bls_result.power)
            features['bls_snr'] = (features['bls_power'] - power_median) / power_std if power_std > 0 else 0

            # Match with known parameters
            if period_hint is not None and period_hint > 0:
                features['period_match'] = abs(features['bls_period'] - period_hint) / period_hint

            if duration_hint is not None and duration_hint > 0:
                features['duration_match'] = abs(features['bls_duration'] - duration_hint) / duration_hint

            if depth_hint is not None and depth_hint > 0:
                features['depth_match'] = abs(features['bls_depth'] - depth_hint) / depth_hint

            features['status'] = 'success'

    except Exception as e:
        features['error'] = str(e)[:100]

    return features

# Main extraction
print(f"\n[5/7] Extracting features from {len(available_samples)} samples...")

results = []
for idx, row in tqdm(available_samples.iterrows(), total=len(available_samples), desc="Extracting"):
    sample_id = row['sample_id']
    tic_id = int(float(row['tic_id']))

    # Find file
    h5_file = LIGHTCURVE_DIR / f"{sample_id}_TIC{tic_id}.h5"

    if not h5_file.exists():
        results.append({
            'sample_id': sample_id,
            'tic_id': tic_id,
            'status': 'file_not_found'
        })
        continue

    # Get hints from dataset
    period_hint = row.get('period', None)
    duration_hint = row.get('duration', None)
    depth_hint = row.get('depth', None)

    # Extract features
    features = extract_bls_features(h5_file, period_hint, duration_hint, depth_hint)
    features['label'] = row.get('label', 0)

    results.append(features)

# Convert to DataFrame
features_df = pd.DataFrame(results)

# Statistics
print(f"\n[6/7] Feature extraction statistics")
status_counts = features_df['status'].value_counts()
for status, count in status_counts.items():
    print(f"  {status}: {count}")

success_rate = status_counts.get('success', 0) / len(features_df) * 100
print(f"  Success rate: {success_rate:.1f}%")

# Check feature quality
if 'bls_power' in features_df.columns:
    successful = features_df[features_df['status'] == 'success']
    print(f"\n  Feature quality (successful extractions):")
    print(f"    BLS power range: {successful['bls_power'].min():.4f} - {successful['bls_power'].max():.4f}")
    print(f"    BLS SNR range: {successful['bls_snr'].min():.2f} - {successful['bls_snr'].max():.2f}")
    print(f"    Period range: {successful['bls_period'].min():.2f} - {successful['bls_period'].max():.2f} days")

    # Check for missing values
    null_counts = successful.isnull().sum()
    if null_counts.sum() > 0:
        print(f"\n  WARNING: Missing values detected:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"    {col}: {count}")

# Save results
output_path = OUTPUT_DIR / 'test_features.csv'
features_df.to_csv(output_path, index=False)

print(f"\n[7/7] Results saved")
print(f"  Output: {output_path}")
print(f"  Total features: {len(features_df)}")
print(f"  Successful: {status_counts.get('success', 0)}")

print("="*70)

if success_rate >= 80:
    print("SUCCESS! Feature extraction passed.")
    print("  Next: Ready for model training")
else:
    print(f"WARNING: Success rate ({success_rate:.1f}%) below target (80%)")
    print("  Review: Check errors and data quality")

print("="*70)
