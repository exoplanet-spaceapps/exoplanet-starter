#!/usr/bin/env python
"""Generate final execution report with statistics."""
import os
import pandas as pd
from pathlib import Path
from datetime import datetime

data_dir = Path(r'C:\Users\tingy\Desktop\dev\exoplanet-starter\data')
results_file = data_dir / 'bls_results.csv'
failed_file = data_dir / 'bls_failed_samples.csv'

print("="*80)
print("BLS BASELINE ANALYSIS - FINAL EXECUTION REPORT")
print("="*80)
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# 1. Processing Summary
# ============================================================================
print("[1] PROCESSING SUMMARY")
print("-" * 80)

total_expected = 11979
successful = 0
failed = 0

if results_file.exists():
    results_df = pd.read_csv(results_file)
    successful = len(results_df)
    print(f"[OK] Successfully processed: {successful} samples")
else:
    print("[PENDING] No results file found")

if failed_file.exists():
    failed_df = pd.read_csv(failed_file)
    failed = len(failed_df)
    print(f"[INFO] Failed: {failed} samples")
else:
    print("[INFO] No failed samples logged")

processed = successful + failed
print(f"\nTotal processed: {processed}/{total_expected} ({processed/total_expected*100:.1f}%)")
print(f"Remaining: {total_expected - processed}")
print()

# ============================================================================
# 2. Results Analysis
# ============================================================================
if results_file.exists() and len(results_df) > 0:
    print("[2] RESULTS ANALYSIS")
    print("-" * 80)

    print("\nLabel Distribution:")
    print(results_df['label'].value_counts().to_dict())

    print("\nBLS Statistics:")
    stats = results_df[['period', 'power', 'duration', 'depth', 'snr']].describe()
    print(stats)

    print("\nSource Distribution:")
    if 'source' in results_df.columns:
        print(results_df['source'].value_counts())

    print()

# ============================================================================
# 3. Failure Analysis
# ============================================================================
if failed_file.exists() and len(failed_df) > 0:
    print("[3] FAILURE ANALYSIS")
    print("-" * 80)

    print("\nFailure Reasons:")
    print(failed_df['reason'].value_counts())
    print()

# ============================================================================
# 4. Output Files
# ============================================================================
print("[4] OUTPUT FILES")
print("-" * 80)

print("\nGenerated Files:")
if results_file.exists():
    size_mb = results_file.stat().st_size / 1024 / 1024
    print(f"  [OK] {results_file} ({size_mb:.2f} MB)")

if failed_file.exists():
    size_mb = failed_file.stat().st_size / 1024 / 1024
    print(f"  [OK] {failed_file} ({size_mb:.2f} MB)")

# Check for checkpoints
import glob
checkpoints = glob.glob(str(data_dir / 'bls_results_checkpoint_*.csv'))
if checkpoints:
    print(f"\nCheckpoint Files: {len(checkpoints)}")
    for cp in checkpoints:
        cp_path = Path(cp)
        size_mb = cp_path.stat().st_size / 1024 / 1024
        print(f"  - {cp_path.name} ({size_mb:.2f} MB)")

print()

# ============================================================================
# 5. Recommendations
# ============================================================================
print("[5] RECOMMENDATIONS")
print("-" * 80)

if processed < total_expected:
    remaining_time_hours = (total_expected - processed) * 15 / 3600  # Assume 15s per sample
    print(f"\nEstimated time to complete remaining {total_expected - processed} samples:")
    print(f"  ~{remaining_time_hours:.1f} hours ({remaining_time_hours/24:.1f} days)")
    print("\nTo continue processing:")
    print("  python scripts/run_bls_analysis_optimized.py")
else:
    print("\n[SUCCESS] All samples processed!")
    print("\nNext Steps:")
    print("  1. Review results in bls_results.csv")
    print("  2. Analyze BLS features for classification")
    print("  3. Proceed to Notebook 03 for model training")

print()
print("="*80)
print("REPORT COMPLETE")
print("="*80)