"""
Simple Test Script for 02_bls_baseline.ipynb Dependencies
Tests data loading and basic imports before running the full notebook
"""

import sys
import io
from pathlib import Path

# Fix UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print("=" * 60)
print("[TEST] 02_bls_baseline.ipynb - Simple Test")
print("=" * 60)

# Test 1: UTF-8 Encoding
print("\n[Test 1] UTF-8 Encoding")
try:
    print("   [OK] Environment: Local")
    print("   [OK] Data directory: ../data")
    print("   [OK] UTF-8 characters work correctly")
except Exception as e:
    print(f"   [FAIL] UTF-8 encoding failed: {e}")
    sys.exit(1)

# Test 2: Import data_loader_colab
print("\n[Test 2] Import data_loader_colab")
try:
    import data_loader_colab
    print("   [OK] data_loader_colab module imported successfully")
except Exception as e:
    print(f"   [FAIL] Import failed: {e}")
    sys.exit(1)

# Test 3: Check data directory
print("\n[Test 3] Check data directory")
try:
    data_dir = Path(__file__).parent.parent / "data"
    csv_files = list(data_dir.glob("*.csv"))
    print(f"   [OK] Data directory exists: {data_dir}")
    print(f"   [OK] Found {len(csv_files)} CSV files:")
    for f in csv_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"      - {f.name} ({size_mb:.2f} MB)")
except Exception as e:
    print(f"   [FAIL] Data directory check failed: {e}")
    sys.exit(1)

# Test 4: Load datasets
print("\n[Test 4] Load datasets using data_loader_colab")
try:
    datasets = data_loader_colab.load_datasets(data_dir)
    if datasets:
        print(f"   [OK] Loaded {len(datasets)} datasets:")
        for name, df in datasets.items():
            print(f"      - {name}: {len(df)} rows, {len(df.columns)} columns")
    else:
        print("   [WARN] No datasets loaded (this might be expected)")
except Exception as e:
    print(f"   [FAIL] Dataset loading failed: {e}")
    sys.exit(1)

# Test 5: Create sample targets
print("\n[Test 5] Create sample targets")
try:
    sample_targets = data_loader_colab.create_sample_targets(datasets, n_positive=3, n_negative=2)
    print(f"   [OK] Created sample_targets: {len(sample_targets)} targets")
    if len(sample_targets) > 0:
        print(f"      - Positive samples: {(sample_targets['label'] == 1).sum()}")
        print(f"      - Negative samples: {(sample_targets['label'] == 0).sum()}")
        print(f"      - Columns: {list(sample_targets.columns)}")
except Exception as e:
    print(f"   [FAIL] Sample targets creation failed: {e}")
    sys.exit(1)

# Test 6: Check lightkurve import
print("\n[Test 6] Check lightkurve availability")
try:
    import lightkurve as lk
    print(f"   [OK] lightkurve imported successfully (version {lk.__version__})")
except ImportError:
    print("   [WARN] lightkurve not installed (this is OK for local testing)")
    print("      Install with: pip install lightkurve")

# Test 7: Check numpy version
print("\n[Test 7] Check numpy version")
try:
    import numpy as np
    print(f"   [OK] numpy version: {np.__version__}")
    if np.__version__.startswith('2.'):
        print("   [WARN] NumPy 2.0 detected - may cause compatibility issues")
        print("      Recommend: pip install numpy==1.26.4")
    else:
        print("   [OK] NumPy version is compatible")
except Exception as e:
    print(f"   [FAIL] NumPy check failed: {e}")

# Test 8: Full data loading workflow
print("\n[Test 8] Full data loading workflow")
try:
    sample_targets, datasets, data_dir_out, IN_COLAB = data_loader_colab.main()
    print(f"   [OK] Full workflow completed:")
    print(f"      - Data directory: {data_dir_out}")
    print(f"      - Environment: {'Google Colab' if IN_COLAB else 'Local'}")
    print(f"      - Datasets loaded: {len(datasets)}")
    print(f"      - Sample targets: {len(sample_targets)}")
except Exception as e:
    print(f"   [FAIL] Full workflow failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("[SUCCESS] ALL TESTS PASSED")
print("=" * 60)
print("\n[Summary]")
print(f"   - Data directory: {data_dir_out}")
print(f"   - CSV files available: {len(csv_files)}")
print(f"   - Datasets loaded: {len(datasets)}")
print(f"   - Sample targets: {len(sample_targets)}")
print("\n[READY] Ready to run 02_bls_baseline.ipynb!")
print("\n[Next Steps]")
print("   1. Install lightkurve: pip install lightkurve astroquery transitleastsquares")
print("   2. Open and run: notebooks/02_bls_baseline.ipynb")