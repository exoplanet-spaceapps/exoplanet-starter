#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test download script - Download 100 samples for testing (FIXED VERSION)"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
warnings.filterwarnings('ignore')

print("="*70)
print("Exoplanet Detection - Test Download (100 samples) - FIXED")
print("="*70)

# Check dependencies
print("\n[1/8] Checking dependencies...")
try:
    import lightkurve as lk
    import h5py
    print(f"  OK: lightkurve {lk.__version__}, numpy {np.__version__}, pandas {pd.__version__}")
except ImportError as e:
    print(f"  ERROR: {e}")
    print("  Install: pip install lightkurve pandas numpy tqdm h5py")
    sys.exit(1)

# Setup paths
BASE_DIR = PROJECT_ROOT
DATA_DIR = BASE_DIR / 'data'
LIGHTCURVE_DIR = DATA_DIR / 'lightcurves'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'

LIGHTCURVE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n[2/8] Paths configured")
print(f"  Working: {BASE_DIR}")
print(f"  Lightcurves: {LIGHTCURVE_DIR}")

# Config
CONFIG = {
    'max_workers': 4,
    'max_retries': 3,
    'timeout': 60,
    'save_interval': 20,
    'test_samples': 11979,  # 改成全量
}

print(f"\n[3/8] Configuration")
for key, val in CONFIG.items():
    print(f"  {key}: {val}")

# Load dataset
print(f"\n[4/8] Loading dataset...")
dataset_path = DATA_DIR / 'supervised_dataset.csv'
if not dataset_path.exists():
    print(f"  ERROR: Dataset not found: {dataset_path}")
    sys.exit(1)

samples_df = pd.read_csv(dataset_path)
samples_df = samples_df.head(CONFIG['test_samples'])

if 'sample_id' not in samples_df.columns:
    samples_df['sample_id'] = [f"SAMPLE_{i:06d}" for i in range(len(samples_df))]
if 'tic_id' not in samples_df.columns:
    if 'tid' in samples_df.columns:
        samples_df['tic_id'] = samples_df['tid']

print(f"  Samples: {len(samples_df)} (Positive: {samples_df['label'].sum()}, Negative: {(~samples_df['label'].astype(bool)).sum()})")

# Download function (FIXED)
def download_lightcurve(row, retries=3):
    sample_id = row['sample_id']
    tic_id = int(float(row['tic_id']))

    result = {'sample_id': sample_id, 'tic_id': tic_id, 'status': 'failed', 'file_path': None, 'n_sectors': 0, 'error': None}

    file_path = LIGHTCURVE_DIR / f"{sample_id}_TIC{tic_id}.h5"
    if file_path.exists():
        result['status'] = 'cached'
        result['file_path'] = str(file_path)
        try:
            with h5py.File(file_path, 'r') as f:
                result['n_sectors'] = f.attrs.get('n_sectors', 0)
        except:
            pass
        return result

    for attempt in range(retries):
        try:
            search_result = lk.search_lightcurve(f"TIC {tic_id}", author='SPOC')
            if search_result is None or len(search_result) == 0:
                result['error'] = 'no_data'
                return result

            lc_collection = search_result.download_all()
            if lc_collection is None or len(lc_collection) == 0:
                result['error'] = 'download_failed'
                return result

            # FIXED: Save as HDF5 instead of pickle
            with h5py.File(file_path, 'w') as f:
                f.attrs['sample_id'] = sample_id
                f.attrs['tic_id'] = tic_id
                f.attrs['n_sectors'] = len(lc_collection)
                f.attrs['download_time'] = datetime.now().isoformat()

                for i, lc in enumerate(lc_collection):
                    grp = f.create_group(f'sector_{i}')
                    # Convert masked arrays to regular numpy arrays
                    grp.create_dataset('time', data=np.array(lc.time.value))
                    grp.create_dataset('flux', data=np.array(lc.flux.value))
                    grp.create_dataset('flux_err', data=np.array(lc.flux_err.value))
                    grp.attrs['sector'] = lc.meta.get('SECTOR', '?')
                    grp.attrs['mission'] = str(lc.meta.get('MISSION', 'TESS'))

            result['status'] = 'success'
            result['file_path'] = str(file_path)
            result['n_sectors'] = len(lc_collection)
            return result
        except Exception as e:
            result['error'] = str(e)[:50]
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return result

# Load checkpoint
def load_checkpoint():
    cp = CHECKPOINT_DIR / 'download_progress.parquet'
    if cp.exists():
        df = pd.read_parquet(cp)
        print(f"  Loaded checkpoint: {len(df)} records")
        return df
    return pd.DataFrame()

def save_checkpoint(df):
    cp = CHECKPOINT_DIR / 'download_progress.parquet'
    df.to_parquet(cp, index=False)

# Main execution
print(f"\n[5/8] Starting download...")
progress_df = load_checkpoint()

if len(progress_df) > 0:
    completed = set(progress_df[progress_df['status'].isin(['success', 'cached'])]['sample_id'])
    remaining = samples_df[~samples_df['sample_id'].isin(completed)]
else:
    remaining = samples_df.copy()

print(f"  Total: {len(samples_df)}, Completed: {len(samples_df)-len(remaining)}, Remaining: {len(remaining)}")

if len(remaining) == 0:
    print("  All samples already downloaded!")
else:
    print(f"  Estimated time: {len(remaining)*5/60/CONFIG['max_workers']:.1f} minutes")

    start_time = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        futures = {executor.submit(download_lightcurve, row, CONFIG['max_retries']): row
                   for _, row in remaining.iterrows()}

        with tqdm(total=len(remaining), desc="Downloading") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

                if len(results) % CONFIG['save_interval'] == 0:
                    temp_df = pd.concat([progress_df, pd.DataFrame(results)], ignore_index=True)
                    save_checkpoint(temp_df)

    if len(results) > 0:
        progress_df = pd.concat([progress_df, pd.DataFrame(results)], ignore_index=True)
        save_checkpoint(progress_df)

    elapsed = time.time() - start_time
    print(f"\n  Complete! Time: {elapsed/60:.1f} min, Avg: {elapsed/len(results):.1f} sec/sample")

# Statistics
print(f"\n[6/8] Final statistics")
status_counts = progress_df['status'].value_counts()
for status, count in status_counts.items():
    print(f"  {status}: {count}")

success_rate = (status_counts.get('success', 0) + status_counts.get('cached', 0)) / len(progress_df) * 100
print(f"  Success rate: {success_rate:.1f}%")

# Verify files
h5_files = list(LIGHTCURVE_DIR.glob('*.h5'))
print(f"\n[7/8] Verification")
print(f"  Files: {len(h5_files)}")

if len(h5_files) > 0:
    total_size = sum(f.stat().st_size for f in h5_files) / 1024 / 1024
    print(f"  Size: {total_size:.1f} MB (avg: {total_size/len(h5_files):.1f} MB/file)")

    if len(h5_files) >= 3:
        samples = np.random.choice(h5_files, min(3, len(h5_files)), replace=False)
        for h5_file in samples:
            try:
                with h5py.File(h5_file, 'r') as f:
                    tic_id = f.attrs['tic_id']
                    n_sectors = f.attrs['n_sectors']
                    sector_0 = f['sector_0']
                    n_points = len(sector_0['time'])
                    print(f"  Sample: TIC{tic_id} - {n_sectors} sectors, {n_points} points")
            except Exception as e:
                print(f"  ERROR: {h5_file.name} - {e}")

# Save report
report = {
    'timestamp': datetime.now().isoformat(),
    'test_samples': CONFIG['test_samples'],
    'downloaded': int(status_counts.get('success', 0) + status_counts.get('cached', 0)),
    'failed': int(status_counts.get('failed', 0)),
    'success_rate': float(success_rate),
    'config': CONFIG,
    'format': 'HDF5'
}

report_path = CHECKPOINT_DIR / 'test_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n[8/8] Report saved: {report_path}")
print("="*70)

if success_rate >= 80:
    print("SUCCESS! Test passed.")
    print("  Next: python scripts/test_features.py")
else:
    print(f"WARNING: Low success rate ({success_rate:.1f}%)")
    print("  Check: network, MAST status, report.json")

print("="*70)
