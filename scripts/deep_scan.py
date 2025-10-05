#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deep scan of all downloaded data - Complete analysis"""

import pandas as pd
import h5py
from pathlib import Path
from datetime import datetime
import json

print("="*70)
print("DEEP SCAN - Complete Download Analysis")
print("="*70)

# 1. Scan checkpoint file
print("\n[1/5] Analyzing checkpoint file...")
checkpoint_path = Path('checkpoints/download_progress.parquet')
if checkpoint_path.exists():
    df = pd.read_parquet(checkpoint_path)
    df_clean = df.drop_duplicates(subset=['sample_id'], keep='last')

    print(f"  Total records: {len(df)} ({len(df) - len(df_clean)} duplicates)")
    print(f"  Unique records: {len(df_clean)}")

    status_counts = df_clean['status'].value_counts()
    print(f"\n  Status breakdown:")
    for status, count in status_counts.items():
        print(f"    {status}: {count} ({count/len(df_clean)*100:.1f}%)")

    # Checkpoint True/False by index
    success_df = df_clean[df_clean['status'].isin(['success', 'cached'])]
    checkpoint_true = len(success_df[success_df.index < 5944])
    checkpoint_false = len(success_df[success_df.index >= 5944])

    print(f"\n  Checkpoint analysis (by index):")
    print(f"    True (index 0-5943): {checkpoint_true}")
    print(f"    False (index 5944+): {checkpoint_false}")
else:
    print("  Checkpoint file not found!")
    df_clean = pd.DataFrame()

# 2. Scan actual HDF5 files
print("\n[2/5] Scanning HDF5 files...")
lightcurve_dir = Path('data/lightcurves')
h5_files = list(lightcurve_dir.glob('*.h5'))

print(f"  Total HDF5 files: {len(h5_files)}")

# Calculate total size
total_size = sum(f.stat().st_size for f in h5_files) / 1024 / 1024 / 1024
print(f"  Total size: {total_size:.2f} GB")
print(f"  Average size: {total_size*1024/len(h5_files):.1f} MB/file")

# 3. Deep analysis of HDF5 metadata
print("\n[3/5] Deep analyzing HDF5 metadata...")
true_samples = []
false_samples = []
corrupted = []
no_metadata = []

for i, h5_file in enumerate(h5_files):
    if (i + 1) % 500 == 0:
        print(f"  Progress: {i+1}/{len(h5_files)}")

    try:
        with h5py.File(h5_file, 'r') as hf:
            sample_id = hf.attrs.get('sample_id', None)

            if sample_id is None:
                no_metadata.append(h5_file.name)
                continue

            # Extract index from sample_id
            if isinstance(sample_id, bytes):
                sample_id = sample_id.decode('utf-8')

            sample_id_str = str(sample_id)

            if 'SAMPLE_' in sample_id_str:
                idx = int(sample_id_str.split('_')[1])

                # Classify by index
                if idx < 5944:
                    true_samples.append({
                        'file': h5_file.name,
                        'sample_id': sample_id_str,
                        'index': idx,
                        'n_sectors': hf.attrs.get('n_sectors', 0)
                    })
                else:
                    false_samples.append({
                        'file': h5_file.name,
                        'sample_id': sample_id_str,
                        'index': idx,
                        'n_sectors': hf.attrs.get('n_sectors', 0)
                    })
    except Exception as e:
        corrupted.append((h5_file.name, str(e)))

print(f"\n  Analysis complete!")
print(f"    True samples: {len(true_samples)}")
print(f"    False samples: {len(false_samples)}")
print(f"    No metadata: {len(no_metadata)}")
print(f"    Corrupted: {len(corrupted)}")

# 4. Quality analysis
print("\n[4/5] Quality analysis...")

if true_samples:
    true_sectors = [s['n_sectors'] for s in true_samples]
    print(f"\n  True samples sector distribution:")
    print(f"    Min sectors: {min(true_sectors)}")
    print(f"    Max sectors: {max(true_sectors)}")
    print(f"    Avg sectors: {sum(true_sectors)/len(true_sectors):.1f}")

if false_samples:
    false_sectors = [s['n_sectors'] for s in false_samples]
    print(f"\n  False samples sector distribution:")
    print(f"    Min sectors: {min(false_sectors)}")
    print(f"    Max sectors: {max(false_sectors)}")
    print(f"    Avg sectors: {sum(false_sectors)/len(false_sectors):.1f}")

# 5. Generate report
print("\n[5/5] Generating report...")

report = {
    'timestamp': datetime.now().isoformat(),
    'total_h5_files': len(h5_files),
    'total_size_gb': round(total_size, 2),
    'checkpoint': {
        'total_records': len(df_clean) if not df_clean.empty else 0,
        'success': len(success_df) if not df_clean.empty else 0,
        'true_by_index': checkpoint_true if not df_clean.empty else 0,
        'false_by_index': checkpoint_false if not df_clean.empty else 0
    },
    'h5_files': {
        'true_samples': len(true_samples),
        'false_samples': len(false_samples),
        'no_metadata': len(no_metadata),
        'corrupted': len(corrupted)
    },
    'quality': {
        'true_avg_sectors': sum(true_sectors)/len(true_sectors) if true_samples else 0,
        'false_avg_sectors': sum(false_sectors)/len(false_sectors) if false_samples else 0
    },
    'balanced_dataset': {
        'max_size': min(len(true_samples), len(false_samples)) * 2,
        'recommended_500_500': 1000
    }
}

report_path = Path('checkpoints/deep_scan_report.json')
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"  Report saved: {report_path}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nTotal HDF5 files: {len(h5_files)} ({total_size:.2f} GB)")
print(f"\nSample distribution:")
print(f"  True (has planet):  {len(true_samples)} samples")
print(f"  False (no planet):  {len(false_samples)} samples")
print(f"  Ratio: {len(true_samples)/len(false_samples):.1f}:1")

print(f"\nBalanced training set options:")
print(f"  Maximum: {min(len(true_samples), len(false_samples))*2} samples")
print(f"  Recommended: 1000 samples (500 + 500)")

print(f"\nData quality:")
if corrupted:
    print(f"  WARNING: {len(corrupted)} corrupted files!")
    for fname, err in corrupted[:5]:
        print(f"    - {fname}: {err}")
else:
    print(f"  All files are valid!")

print("\n" + "="*70)
print("Next steps:")
print("  1. Extract features from all HDF5 files")
print("  2. Prepare balanced dataset (500 True + 500 False)")
print("  3. Train XGBoost model")
print("="*70)
