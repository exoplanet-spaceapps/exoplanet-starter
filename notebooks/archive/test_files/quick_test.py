"""Quick test of data_loader_colab without hanging"""
import sys
from pathlib import Path

# Test 1: Import
print("[Test 1] Import data_loader_colab")
import data_loader_colab
print("   [OK] Module imported")

# Test 2: Check data directory
print("\n[Test 2] Check data directory")
data_dir = Path(__file__).parent.parent / "data"
print(f"   [OK] Data directory: {data_dir}")
print(f"   [OK] Exists: {data_dir.exists()}")

# Test 3: List CSV files
print("\n[Test 3] List CSV files")
csv_files = list(data_dir.glob("*.csv"))
print(f"   [OK] Found {len(csv_files)} CSV files")
for f in csv_files[:5]:  # Show first 5
    print(f"   - {f.name}")

# Test 4: Load datasets function (not main)
print("\n[Test 4] Load datasets")
datasets = data_loader_colab.load_datasets(data_dir)
print(f"   [OK] Loaded {len(datasets)} datasets")
for name in datasets.keys():
    print(f"   - {name}: {len(datasets[name])} rows")

# Test 5: Create sample targets
print("\n[Test 5] Create sample targets")
sample_targets = data_loader_colab.create_sample_targets(datasets, n_positive=2, n_negative=1)
print(f"   [OK] Created {len(sample_targets)} sample targets")

print("\n[SUCCESS] All basic tests passed!")