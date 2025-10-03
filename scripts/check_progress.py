#!/usr/bin/env python
"""Check progress of BLS analysis."""
import os
import pandas as pd
from pathlib import Path

data_dir = Path(r'C:\Users\tingy\Desktop\dev\exoplanet-starter\data')

# Check if results file exists
results_file = data_dir / 'bls_results.csv'
failed_file = data_dir / 'bls_failed_samples.csv'

print("BLS Analysis Progress Check")
print("="*80)

if results_file.exists():
    results_df = pd.read_csv(results_file)
    print(f"[OK] Results file exists: {results_file}")
    print(f"  Samples processed: {len(results_df)}")
    print(f"  Columns: {list(results_df.columns)}")
    if len(results_df) > 0:
        print(f"\n  Sample results:")
        print(results_df.head())
        print(f"\n  Label distribution:")
        print(results_df['label'].value_counts())
else:
    print("[PENDING] Results file not yet created")

print()

if failed_file.exists():
    failed_df = pd.read_csv(failed_file)
    print(f"[INFO] Failed samples file exists: {failed_file}")
    print(f"  Failed samples: {len(failed_df)}")
    if len(failed_df) > 0:
        print(f"\n  Failure reasons:")
        print(failed_df['reason'].value_counts())
else:
    print("[INFO] No failed samples file yet")

# Check total expected
total_expected = 11979
if results_file.exists():
    successful = len(results_df)
    failed = len(failed_df) if failed_file.exists() else 0
    processed = successful + failed
    print()
    print(f"Total Expected: {total_expected}")
    print(f"Successfully Processed: {successful} ({successful/total_expected*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total_expected*100:.1f}%)")
    print(f"Total Processed: {processed} ({processed/total_expected*100:.1f}%)")
    print(f"Remaining: {total_expected - processed}")