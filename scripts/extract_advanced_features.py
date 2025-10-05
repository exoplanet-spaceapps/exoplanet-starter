#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced Feature Extraction - Time Series + Wavelet Features
Improves upon basic BLS features with temporal and frequency domain features
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py

warnings.filterwarnings('ignore')

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

print("="*70)
print("Advanced Feature Extraction - Time Series + Wavelet")
print("="*70)

# Check dependencies
print("\n[1/7] Checking dependencies...")
try:
    from astropy.timeseries import BoxLeastSquares
    from scipy import stats, signal
    from scipy.fft import fft
    import pywt  # PyWavelets for wavelet transform
    print(f"  OK: astropy, scipy, pywt")
except ImportError as e:
    print(f"  ERROR: {e}")
    print("  Install: pip install pywt")
    sys.exit(1)

# Paths
DATA_DIR = PROJECT_ROOT / 'data'
LIGHTCURVE_DIR = DATA_DIR / 'lightcurves'
OUTPUT_DIR = DATA_DIR

# Load dataset
print(f"\n[2/7] Loading dataset...")
features_path = DATA_DIR / 'balanced_features.csv'
df = pd.read_csv(features_path)
df_success = df[df['status'] == 'success'].copy()

print(f"  Samples: {len(df_success)}")

# Advanced feature extraction
def extract_advanced_features(h5_file_path, sample_id, tic_id, label):
    """Extract advanced time series and wavelet features"""

    features = {
        'sample_id': sample_id,
        'tic_id': tic_id,
        'label': label,

        # Basic stats (from previous)
        'flux_mean': np.nan,
        'flux_std': np.nan,
        'flux_median': np.nan,
        'flux_mad': np.nan,
        'flux_skew': np.nan,
        'flux_kurt': np.nan,

        # BLS features (from previous)
        'bls_period': np.nan,
        'bls_duration': np.nan,
        'bls_depth': np.nan,
        'bls_power': np.nan,
        'bls_snr': np.nan,

        # NEW: Time series features
        'autocorr_lag1': np.nan,
        'autocorr_lag5': np.nan,
        'trend_slope': np.nan,
        'variability': np.nan,

        # NEW: Frequency domain features
        'fft_peak_freq': np.nan,
        'fft_peak_power': np.nan,
        'spectral_entropy': np.nan,

        # NEW: Wavelet features
        'wavelet_energy': np.nan,
        'wavelet_entropy': np.nan,
        'wavelet_var': np.nan,

        'status': 'failed',
        'error': None
    }

    try:
        with h5py.File(h5_file_path, 'r') as f:
            n_sectors = f.attrs.get('n_sectors', 0)

            # Combine all sectors
            time_list = []
            flux_list = []

            for i in range(n_sectors):
                sector = f[f'sector_{i}']
                time = np.array(sector['time'][:])
                flux = np.array(sector['flux'][:])

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

            # Normalize
            flux_median = np.median(flux)
            flux = flux / flux_median

            # === Basic Statistics ===
            features['flux_mean'] = np.mean(flux)
            features['flux_std'] = np.std(flux)
            features['flux_median'] = np.median(flux)
            features['flux_mad'] = np.median(np.abs(flux - np.median(flux)))
            features['flux_skew'] = stats.skew(flux)
            features['flux_kurt'] = stats.kurtosis(flux)

            # === BLS Analysis ===
            periods = np.linspace(0.5, 15.0, 2000)
            durations = np.linspace(0.05, 0.3, 10)

            bls = BoxLeastSquares(time, flux)
            bls_result = bls.power(periods, durations)

            best_idx = np.argmax(bls_result.power)
            features['bls_period'] = bls_result.period[best_idx]
            features['bls_duration'] = bls_result.duration[best_idx]
            features['bls_depth'] = bls_result.depth[best_idx]
            features['bls_power'] = bls_result.power[best_idx]

            power_median = np.median(bls_result.power)
            power_std = np.std(bls_result.power)
            features['bls_snr'] = (features['bls_power'] - power_median) / power_std if power_std > 0 else 0

            # === NEW: Time Series Features ===

            # Autocorrelation
            if len(flux) > 5:
                autocorr = np.correlate(flux - flux.mean(), flux - flux.mean(), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]

                if len(autocorr) > 1:
                    features['autocorr_lag1'] = autocorr[1]
                if len(autocorr) > 5:
                    features['autocorr_lag5'] = autocorr[5]

            # Trend (linear regression slope)
            if len(time) > 2:
                slope, _, _, _, _ = stats.linregress(time, flux)
                features['trend_slope'] = slope

            # Variability (coefficient of variation)
            if features['flux_mean'] != 0:
                features['variability'] = features['flux_std'] / abs(features['flux_mean'])

            # === NEW: Frequency Domain Features (FFT) ===

            if len(flux) > 10:
                # FFT
                fft_vals = np.abs(fft(flux))
                fft_freqs = np.fft.fftfreq(len(flux), d=np.median(np.diff(time)))

                # Positive frequencies only
                pos_mask = fft_freqs > 0
                fft_vals = fft_vals[pos_mask]
                fft_freqs = fft_freqs[pos_mask]

                if len(fft_vals) > 0:
                    peak_idx = np.argmax(fft_vals)
                    features['fft_peak_freq'] = fft_freqs[peak_idx]
                    features['fft_peak_power'] = fft_vals[peak_idx]

                    # Spectral entropy
                    psd = fft_vals ** 2
                    psd_norm = psd / psd.sum()
                    psd_norm = psd_norm[psd_norm > 0]
                    features['spectral_entropy'] = -np.sum(psd_norm * np.log(psd_norm))

            # === NEW: Wavelet Features ===

            if len(flux) > 20:
                # Discrete Wavelet Transform (DWT)
                coeffs = pywt.wavedec(flux, 'db4', level=3)

                # Wavelet energy
                energies = [np.sum(c**2) for c in coeffs]
                features['wavelet_energy'] = np.sum(energies)

                # Wavelet entropy
                energies_norm = np.array(energies) / np.sum(energies)
                energies_norm = energies_norm[energies_norm > 0]
                features['wavelet_entropy'] = -np.sum(energies_norm * np.log(energies_norm))

                # Wavelet variance (detail coefficients)
                features['wavelet_var'] = np.var(np.concatenate(coeffs[1:]))

            features['status'] = 'success'

    except Exception as e:
        features['error'] = str(e)[:100]

    return features

# Extract features
print(f"\n[3/7] Extracting advanced features from {len(df_success)} samples...")

results = []
for idx, row in tqdm(df_success.iterrows(), total=len(df_success), desc="Processing"):
    sample_id = row['sample_id']
    tic_id = row['tic_id']
    label = row['label']

    h5_file = LIGHTCURVE_DIR / f"{sample_id}_TIC{int(tic_id)}.h5"

    if h5_file.exists():
        features = extract_advanced_features(h5_file, sample_id, tic_id, label)
        results.append(features)

# Convert to DataFrame
advanced_features_df = pd.DataFrame(results)

# Statistics
print(f"\n[4/7] Feature extraction statistics")
status_counts = advanced_features_df['status'].value_counts()
for status, count in status_counts.items():
    print(f"  {status}: {count}")

success_rate = status_counts.get('success', 0) / len(advanced_features_df) * 100
print(f"  Success rate: {success_rate:.1f}%")

# Feature summary
successful = advanced_features_df[advanced_features_df['status'] == 'success']
if len(successful) > 0:
    print(f"\n[5/7] Feature summary (successful extractions):")

    new_features = [
        'autocorr_lag1', 'autocorr_lag5', 'trend_slope', 'variability',
        'fft_peak_freq', 'fft_peak_power', 'spectral_entropy',
        'wavelet_energy', 'wavelet_entropy', 'wavelet_var'
    ]

    print(f"\n  New features added: {len(new_features)}")
    for feat in new_features:
        if feat in successful.columns:
            valid_count = successful[feat].notna().sum()
            print(f"    {feat}: {valid_count}/{len(successful)} valid")

# Save results
output_path = OUTPUT_DIR / 'advanced_features.csv'
advanced_features_df.to_csv(output_path, index=False)

print(f"\n[6/7] Results saved")
print(f"  Output: {output_path}")
print(f"  Total features: {len(advanced_features_df.columns) - 3}")  # Exclude sample_id, status, error

# Comparison
print(f"\n[7/7] Feature comparison:")
print(f"  Basic features (balanced_features.csv): 11 features")
print(f"  Advanced features (advanced_features.csv): {len(new_features) + 11} features")
print(f"  New features added: {len(new_features)}")

print("="*70)

if success_rate >= 90:
    print("SUCCESS! Advanced feature extraction complete.")
    print(f"\nNext: Train XGBoost with advanced features")
    print(f"  python scripts/train_advanced_model.py")
else:
    print(f"WARNING: Success rate ({success_rate:.1f}%) below target (90%)")

print("="*70)
