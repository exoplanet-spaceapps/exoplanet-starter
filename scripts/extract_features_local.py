#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Feature Extraction Script
================================
Extract BLS features from all 11,979 TOI samples locally.
Results will be uploaded to repo for Colab training.

Usage:
    python scripts/extract_features_local.py

Estimated time: 4-8 hours
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Fix Windows encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import numpy as np
import pandas as pd
from tqdm import tqdm

# Astronomy
import lightkurve as lk
from astropy.timeseries import BoxLeastSquares

warnings.filterwarnings('ignore')
np.random.seed(42)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Configuration
CONFIG = {
    'batch_size': 50,
    'max_retries': 3,
    'save_interval': 100,
    'checkpoint_file': CHECKPOINT_DIR / 'features_local_checkpoint.parquet',
    'final_output': DATA_DIR / 'features_extracted_full.parquet',
}

print("=" * 70)
print("LOCAL FEATURE EXTRACTION")
print("=" * 70)
print(f"Base directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Checkpoint: {CONFIG['checkpoint_file']}")
print(f"Output: {CONFIG['final_output']}")
print()


def extract_features_from_lightcurve(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    duration: float,
    depth: float
) -> Dict[str, float]:
    """Extract BLS features from light curve."""
    features = {}

    # Basic statistics
    features['flux_mean'] = float(np.nanmean(flux))
    features['flux_std'] = float(np.nanstd(flux))
    features['flux_median'] = float(np.nanmedian(flux))
    features['flux_mad'] = float(np.nanmedian(np.abs(flux - np.nanmedian(flux))))
    features['n_points'] = int(len(time))
    features['time_span'] = float(time[-1] - time[0])

    # Input parameters
    features['input_period'] = float(period)
    features['input_duration'] = float(duration)
    features['input_depth'] = float(depth)

    # BLS analysis
    try:
        bls = BoxLeastSquares(time, flux)
        periods = np.linspace(0.5, 15.0, 2000)
        durations = np.linspace(0.05, 0.5, 10)
        bls_result = bls.power(periods, durations)

        max_idx = np.argmax(bls_result.power)
        features['bls_power'] = float(bls_result.power[max_idx])
        features['bls_period'] = float(bls_result.period[max_idx])
        features['bls_duration'] = float(bls_result.duration[max_idx])
        features['bls_depth'] = float(bls_result.depth[max_idx])
        features['bls_snr'] = float(bls_result.depth_snr[max_idx])
    except Exception as e:
        features.update({
            'bls_power': 0.0,
            'bls_period': float(period),
            'bls_duration': float(duration),
            'bls_depth': float(depth),
            'bls_snr': 0.0
        })

    return features


def download_and_extract_single_sample(row: pd.Series, retries: int = 3) -> Optional[Dict]:
    """Download light curve and extract features for single sample."""
    for attempt in range(retries):
        try:
            tic_id = int(float(row['tic_id']))

            # Search for light curves
            search_result = lk.search_lightcurve(f"TIC {tic_id}", author='SPOC')
            if search_result is None or len(search_result) == 0:
                return None

            # Download first available
            lc_collection = search_result.download_all()
            if lc_collection is None or len(lc_collection) == 0:
                return None

            lc = lc_collection[0].remove_nans().normalize()

            # Extract features
            features = extract_features_from_lightcurve(
                lc.time.value, lc.flux.value,
                row.get('period', 1.0),
                row.get('duration', 0.1),
                row.get('depth', 0.01)
            )

            features['sample_id'] = row['sample_id']
            features['tic_id'] = int(tic_id)
            features['label'] = int(row['label'])

            return features

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None

    return None


def load_checkpoint() -> pd.DataFrame:
    """Load features from checkpoint."""
    if CONFIG['checkpoint_file'].exists():
        df = pd.read_parquet(CONFIG['checkpoint_file'])
        print(f"📂 Loaded checkpoint: {len(df):,} samples")
        return df
    return pd.DataFrame()


def save_checkpoint(features_df: pd.DataFrame):
    """Save features to checkpoint."""
    features_df.to_parquet(CONFIG['checkpoint_file'], index=False)
    print(f"💾 Checkpoint saved: {len(features_df):,} samples")


def main():
    # Load dataset
    supervised_path = DATA_DIR / 'supervised_dataset.csv'
    if not supervised_path.exists():
        print(f"❌ Dataset not found: {supervised_path}")
        return 1

    samples_df = pd.read_csv(supervised_path)

    # Add sample_id if not present
    if 'sample_id' not in samples_df.columns:
        samples_df['sample_id'] = [f"SAMPLE_{i:06d}" for i in range(len(samples_df))]

    # Map tid to tic_id
    if 'tic_id' not in samples_df.columns:
        if 'tid' in samples_df.columns:
            samples_df['tic_id'] = samples_df['tid']
        elif 'target_id' in samples_df.columns:
            samples_df['tic_id'] = samples_df['target_id']

    print(f"✅ Loaded dataset: {len(samples_df):,} samples")
    print(f"   Positive: {samples_df['label'].sum():,} ({samples_df['label'].mean():.1%})")
    print(f"   Negative: {(~samples_df['label'].astype(bool)).sum():,}")
    print()

    # Load checkpoint
    features_df = load_checkpoint()

    # Determine remaining samples
    if len(features_df) > 0:
        processed_ids = set(features_df['sample_id'].values)
        remaining_samples = samples_df[~samples_df['sample_id'].isin(processed_ids)]
    else:
        remaining_samples = samples_df.copy()

    print(f"📊 Progress:")
    print(f"   Total: {len(samples_df):,}")
    print(f"   Processed: {len(features_df):,}")
    print(f"   Remaining: {len(remaining_samples):,}")
    print()

    if len(remaining_samples) == 0:
        print("✅ All samples already processed!")
        return 0

    # Estimate time
    samples_per_hour = 1200  # ~3 sec per sample
    estimated_hours = len(remaining_samples) / samples_per_hour
    print(f"Estimated time: {estimated_hours:.1f} hours")
    print(f"   ({len(remaining_samples):,} samples x ~3 sec/sample)")
    print()

    print("=" * 70)
    print("STARTING FEATURE EXTRACTION")
    print("=" * 70)
    print()

    start_time = time.time()
    new_features = []
    failed_count = 0

    # Process in batches
    for batch_start in range(0, len(remaining_samples), CONFIG['batch_size']):
        batch_end = min(batch_start + CONFIG['batch_size'], len(remaining_samples))
        batch = remaining_samples.iloc[batch_start:batch_end]

        batch_num = batch_start // CONFIG['batch_size'] + 1
        total_batches = (len(remaining_samples) + CONFIG['batch_size'] - 1) // CONFIG['batch_size']

        print(f"\nBatch {batch_num}/{total_batches}: Samples {batch_start+1}-{batch_end}")

        # Process batch
        for _, row in tqdm(batch.iterrows(), total=len(batch), desc="  Processing"):
            result = download_and_extract_single_sample(row, CONFIG['max_retries'])

            if result is not None:
                new_features.append(result)
            else:
                failed_count += 1

            # Save checkpoint
            if len(new_features) % CONFIG['save_interval'] == 0 and len(new_features) > 0:
                temp_df = pd.concat([features_df, pd.DataFrame(new_features)], ignore_index=True)
                save_checkpoint(temp_df)

        # Progress stats
        elapsed = time.time() - start_time
        processed = len(new_features) + len(features_df)
        total_processed = len(new_features) + failed_count
        success_rate = (len(new_features) / total_processed * 100) if total_processed > 0 else 0

        samples_per_sec = (batch_end) / elapsed if elapsed > 0 else 0
        remaining_time = (len(remaining_samples) - batch_end) / samples_per_sec if samples_per_sec > 0 else 0

        print(f"  Success: {len(new_features):,} | Failed: {failed_count:,} | Rate: {success_rate:.1f}%")
        print(f"  Elapsed: {elapsed/60:.1f} min | Remaining: ~{remaining_time/60:.1f} min")
        print(f"  Progress: {processed:,}/{len(samples_df):,} ({processed/len(samples_df)*100:.1f}%)")

    # Final save
    if len(new_features) > 0:
        features_df = pd.concat([features_df, pd.DataFrame(new_features)], ignore_index=True)
        save_checkpoint(features_df)

    # Save final output
    features_df.to_parquet(CONFIG['final_output'], index=False)

    # Summary
    total_time = time.time() - start_time

    print()
    print("=" * 70)
    print("FEATURE EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"   Total samples processed: {len(features_df):,}")
    print(f"   Success rate: {len(features_df)/(len(features_df)+failed_count)*100:.1f}%")
    print(f"   Total time: {total_time/3600:.2f} hours")
    print(f"   Average: {total_time/len(features_df):.2f} sec/sample")
    print()
    print(f"Saved to: {CONFIG['final_output']}")
    print(f"   File size: {CONFIG['final_output'].stat().st_size / (1024**2):.1f} MB")
    print()
    print("Next steps:")
    print("   1. Commit and push to repo")
    print("   2. Run Cell 7-9 in Colab to train model")

    return 0


if __name__ == "__main__":
    sys.exit(main())
