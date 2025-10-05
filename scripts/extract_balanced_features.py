#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract features from balanced dataset (500 True + 500 False)"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
warnings.filterwarnings('ignore')

print("="*70)
print("Feature Extraction - Balanced Dataset (500 + 500)")
print("="*70)

# Check dependencies
print("\n[1/8] Checking dependencies...")
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
OUTPUT_DIR = DATA_DIR
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n[2/8] Paths configured")
print(f"  Lightcurves: {LIGHTCURVE_DIR}")
print(f"  Output: {OUTPUT_DIR}")

# BLS Configuration
CONFIG = {
    'bls_periods': 2000,
    'period_min': 0.5,
    'period_max': 15.0,
    'duration_min': 0.05,
    'duration_max': 0.3,
    'n_durations': 10,
}

print(f"\n[3/8] BLS Configuration")
for key, val in CONFIG.items():
    print(f"  {key}: {val}")

# Scan HDF5 files and classify
print(f"\n[4/8] Scanning HDF5 files...")
h5_files = list(LIGHTCURVE_DIR.glob('*.h5'))
print(f"  Total files: {len(h5_files)}")

true_samples = []
false_samples = []

for h5_file in tqdm(h5_files, desc="Classifying"):
    try:
        with h5py.File(h5_file, 'r') as hf:
            sample_id = hf.attrs.get('sample_id', None)
            tic_id = hf.attrs.get('tic_id', 0)
            n_sectors = hf.attrs.get('n_sectors', 0)

            if sample_id is None:
                continue

            # Extract index from sample_id
            if isinstance(sample_id, bytes):
                sample_id = sample_id.decode('utf-8')

            sample_id_str = str(sample_id)

            if 'SAMPLE_' in sample_id_str:
                idx = int(sample_id_str.split('_')[1])

                file_info = {
                    'file_path': h5_file,
                    'sample_id': sample_id_str,
                    'tic_id': tic_id,
                    'n_sectors': n_sectors,
                    'index': idx
                }

                # Classify by index
                if idx < 5944:
                    true_samples.append(file_info)
                else:
                    false_samples.append(file_info)
    except Exception as e:
        continue

print(f"\n  Classification complete:")
print(f"    True samples: {len(true_samples)}")
print(f"    False samples: {len(false_samples)}")

# Random selection
print(f"\n[5/8] Selecting balanced dataset...")
random.seed(42)

n_true = min(500, len(true_samples))
n_false = min(500, len(false_samples))

selected_true = random.sample(true_samples, n_true)
selected_false = random.sample(false_samples, n_false)

selected_samples = selected_true + selected_false
random.shuffle(selected_samples)

print(f"  Selected: {n_true} True + {n_false} False = {len(selected_samples)} total")

# Feature extraction function
def extract_bls_features(h5_file_path, sample_id, tic_id, label):
    """Extract BLS features from HDF5 file"""

    features = {
        'sample_id': sample_id,
        'tic_id': tic_id,
        'label': label,
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
        'status': 'failed',
        'error': None
    }

    try:
        with h5py.File(h5_file_path, 'r') as f:
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

            features['status'] = 'success'

    except Exception as e:
        features['error'] = str(e)[:100]

    return features

# Extract features
print(f"\n[6/8] Extracting features from {len(selected_samples)} samples...")

results = []
for sample_info in tqdm(selected_samples, desc="Extracting"):
    # Determine label
    label = 1 if sample_info['index'] < 5944 else 0

    features = extract_bls_features(
        sample_info['file_path'],
        sample_info['sample_id'],
        sample_info['tic_id'],
        label
    )

    results.append(features)

# Convert to DataFrame
features_df = pd.DataFrame(results)

# Statistics
print(f"\n[7/8] Feature extraction statistics")
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

    # Label distribution
    label_counts = successful['label'].value_counts()
    print(f"\n  Label distribution:")
    print(f"    True (label=1): {label_counts.get(1, 0)}")
    print(f"    False (label=0): {label_counts.get(0, 0)}")

    # Check for missing values
    null_counts = successful.isnull().sum()
    if null_counts.sum() > 0:
        print(f"\n  WARNING: Missing values detected:")
        for col, count in null_counts[null_counts > 0].items():
            print(f"    {col}: {count}")

# Save results
output_path = OUTPUT_DIR / 'balanced_features.csv'
features_df.to_csv(output_path, index=False)

print(f"\n[8/8] Results saved")
print(f"  Output: {output_path}")
print(f"  Total features: {len(features_df)}")
print(f"  Successful: {status_counts.get('success', 0)}")

print("="*70)

if success_rate >= 90:
    print("SUCCESS! Feature extraction complete.")
    print(f"  Next: Train XGBoost model with {len(features_df)} balanced samples")
else:
    print(f"WARNING: Success rate ({success_rate:.1f}%) below target (90%)")
    print("  Review: Check errors and data quality")

print("="*70)
