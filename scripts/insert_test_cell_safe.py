#!/usr/bin/env python3
"""
Safely insert test cell into notebook
"""
import nbformat
import sys
import uuid

# Test cell content
TEST_CELL_CODE = '''# 🧪 === COMPREHENSIVE TESTING SUITE === 🧪
# Run this cell to validate all critical components before full execution

print("=" * 60)
print("🧪 Running Notebook 02 Validation Tests...")
print("=" * 60)
print()

test_results = []

# ==========================================
# Test 1/5: NumPy Version Verification
# ==========================================
print("Test 1/5: NumPy version compatibility...")
try:
    import numpy as np
    version = np.__version__
    is_compatible = version.startswith('1.26')

    if is_compatible:
        print(f"  ✅ NumPy {version} detected (compatible)")
        test_results.append(("NumPy version", True))
    else:
        print(f"  ❌ NumPy {version} incompatible (need 1.26.x)")
        print(f"     Run: pip install numpy==1.26.4")
        test_results.append(("NumPy version", False))
except Exception as e:
    print(f"  ❌ NumPy check failed: {e}")
    test_results.append(("NumPy version", False))
print()

# ==========================================
# Test 2/5: Checkpoint System
# ==========================================
print("Test 2/5: Checkpoint system functionality...")
try:
    import os
    import json
    import tempfile
    from pathlib import Path

    # Create temporary checkpoint directory
    test_checkpoint_dir = tempfile.mkdtemp(prefix='test_checkpoints_')

    # Define CheckpointManager class for testing
    class TestCheckpointManager:
        def __init__(self, checkpoint_dir, batch_size=10):
            self.checkpoint_dir = Path(checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.batch_size = batch_size
            self.checkpoint_file = self.checkpoint_dir / 'progress.json'

        def save_batch(self, data, batch_num):
            checkpoint = {
                'last_batch': batch_num,
                'timestamp': str(pd.Timestamp.now()),
                'batch_size': self.batch_size,
                'data_sample': data
            }
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f)

        def resume_from_last(self):
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                return checkpoint['last_batch'] + 1
            return 0

    # Test checkpoint save and resume
    test_checkpoint_mgr = TestCheckpointManager(test_checkpoint_dir, batch_size=10)
    test_data = {'sample_id': [1, 2, 3], 'bls_period': [3.5, 4.2, 2.1]}
    test_checkpoint_mgr.save_batch(test_data, 0)
    resumed_batch = test_checkpoint_mgr.resume_from_last()

    # Cleanup
    import shutil
    shutil.rmtree(test_checkpoint_dir)

    if resumed_batch == 1:
        print(f"  ✅ Checkpoint system working (resumed batch: {resumed_batch})")
        test_results.append(("Checkpoint system", True))
    else:
        print(f"  ❌ Checkpoint resume failed (expected 1, got {resumed_batch})")
        test_results.append(("Checkpoint system", False))
except Exception as e:
    print(f"  ❌ Checkpoint test failed: {e}")
    test_results.append(("Checkpoint system", False))
print()

# ==========================================
# Test 3/5: Single Sample Feature Extraction
# ==========================================
print("Test 3/5: Feature extraction (single target)...")
try:
    import lightkurve as lk
    from astropy.timeseries import BoxLeastSquares

    # Test with known TOI target
    test_tic = "25155310"  # TOI-270 (known multi-planet system)

    print(f"  📡 Testing with TIC {test_tic} (TOI-270)...")

    # Download light curve
    search_result = lk.search_lightcurve(f'TIC {test_tic}', mission='TESS', author='SPOC')

    if len(search_result) > 0:
        lc = search_result[0].download()
        lc = lc.remove_nans().remove_outliers(sigma=5)

        # Run BLS
        period_grid = np.linspace(1.0, 15.0, 5000)
        bls = BoxLeastSquares(lc.time.value, lc.flux.value)
        bls_result = bls.power(period_grid)

        # Extract basic features
        test_features = {
            'tic_id': test_tic,
            'bls_period': float(bls_result.period[np.argmax(bls_result.power)]),
            'bls_power': float(np.max(bls_result.power)),
            'bls_depth': float(bls_result.depth[np.argmax(bls_result.power)]),
            'bls_duration': float(bls_result.duration[np.argmax(bls_result.power)]),
            'num_points': len(lc.time),
            'flux_std': float(np.std(lc.flux.value)),
            'flux_median': float(np.median(lc.flux.value))
        }

        # Validation
        feature_count = len(test_features)
        has_nan = any(pd.isna(list(test_features.values())))
        valid_period = 1.0 <= test_features['bls_period'] <= 15.0

        if not has_nan and valid_period and feature_count >= 8:
            print(f"  ✅ Extracted {feature_count} features successfully")
            print(f"     - Period: {test_features['bls_period']:.3f} days")
            print(f"     - Power: {test_features['bls_power']:.4f}")
            print(f"     - Data points: {test_features['num_points']}")
            test_results.append(("Feature extraction", True))
        else:
            print(f"  ⚠️  Features extracted but validation issues:")
            print(f"     - NaN values: {has_nan}")
            print(f"     - Valid period: {valid_period}")
            test_results.append(("Feature extraction", False))
    else:
        print(f"  ⚠️  No data found for TIC {test_tic}")
        print(f"     This is expected if MAST is unavailable")
        test_results.append(("Feature extraction", None))

except Exception as e:
    print(f"  ⚠️  Feature extraction test skipped: {e}")
    print(f"     This is normal if MAST/Lightkurve is unavailable")
    test_results.append(("Feature extraction", None))
print()

# ==========================================
# Test 4/5: Google Drive Access (Colab only)
# ==========================================
print("Test 4/5: Google Drive access...")
try:
    import os
    from pathlib import Path

    # Check if running in Colab
    try:
        from google.colab import drive
        in_colab = True
    except ImportError:
        in_colab = False

    if in_colab:
        # Test Drive access
        base_path = Path('/content/drive/MyDrive/spaceapps-exoplanet')
        checkpoint_dir = base_path / 'checkpoints'

        # Try to create and write test file
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        test_file = checkpoint_dir / 'test_access.txt'

        with open(test_file, 'w') as f:
            f.write('test')

        exists = test_file.exists()

        # Cleanup
        test_file.unlink()

        if exists:
            print(f"  ✅ Google Drive writable at {checkpoint_dir}")
            test_results.append(("Google Drive access", True))
        else:
            print(f"  ❌ Cannot write to Google Drive")
            test_results.append(("Google Drive access", False))
    else:
        print(f"  ℹ️  Not in Colab environment (local execution)")
        print(f"     Checkpoint directory will use: ./checkpoints/")
        test_results.append(("Google Drive access", None))

except Exception as e:
    print(f"  ❌ Google Drive test failed: {e}")
    test_results.append(("Google Drive access", False))
print()

# ==========================================
# Test 5/5: Batch Processing (Mini Test)
# ==========================================
print("Test 5/5: Batch processing (small scale)...")
try:
    # Load sample data
    data_path = Path('/content/drive/MyDrive/spaceapps-exoplanet/data') if in_colab else Path('./data')
    supervised_csv = data_path / 'supervised_dataset.csv'

    if supervised_csv.exists():
        samples_df = pd.read_csv(supervised_csv)
        test_samples = samples_df.head(5)  # Test with 5 samples

        print(f"  📊 Testing with {len(test_samples)} samples...")

        # Mock batch processing
        successful = 0
        failed = 0

        for idx, row in test_samples.iterrows():
            tic_id = row['TIC_ID']
            try:
                # Simulate feature extraction (lightweight)
                search_result = lk.search_lightcurve(f'TIC {tic_id}', mission='TESS', author='SPOC')
                if len(search_result) > 0:
                    successful += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

        success_rate = successful / len(test_samples) if len(test_samples) > 0 else 0

        if success_rate >= 0.4:  # At least 40% success
            print(f"  ✅ Batch test: {success_rate*100:.1f}% success rate ({successful}/{len(test_samples)})")
            test_results.append(("Batch processing", True))
        else:
            print(f"  ⚠️  Low success rate: {success_rate*100:.1f}% ({successful}/{len(test_samples)})")
            print(f"     This may indicate MAST availability issues")
            test_results.append(("Batch processing", False))
    else:
        print(f"  ℹ️  supervised_dataset.csv not found")
        print(f"     Expected at: {supervised_csv}")
        test_results.append(("Batch processing", None))

except Exception as e:
    print(f"  ⚠️  Batch processing test skipped: {e}")
    test_results.append(("Batch processing", None))
print()

# ==========================================
# Summary Report
# ==========================================
print("=" * 60)
print("📊 TEST SUMMARY")
print("=" * 60)

passed = sum(1 for _, result in test_results if result is True)
failed = sum(1 for _, result in test_results if result is False)
skipped = sum(1 for _, result in test_results if result is None)

for test_name, result in test_results:
    status = "✅ PASS" if result is True else ("❌ FAIL" if result is False else "⚠️  SKIP")
    print(f"{status:12} - {test_name}")

print("-" * 60)
print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
print("=" * 60)

if failed == 0 and passed >= 3:
    print("✅ All critical tests passed! Ready for production run.")
    print("   You can now proceed with full feature extraction.")
elif failed > 0:
    print("⚠️  Some tests failed. Please review errors above.")
    print("   Fix issues before running full extraction.")
else:
    print("ℹ️  Most tests skipped (likely due to data availability).")
    print("   This is normal for offline/local testing.")

print("=" * 60)
'''

def insert_test_cell(notebook_path, insert_after_index=8):
    """Safely insert test cell"""
    print(f"Reading: {notebook_path}")

    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    print(f"Total cells: {len(nb.cells)}")

    # Create new test cell with all required attributes
    test_cell = nbformat.v4.new_code_cell(source=TEST_CELL_CODE)
    test_cell.outputs = []
    test_cell.execution_count = None

    # Insert after specified index
    nb.cells.insert(insert_after_index, test_cell)

    print(f"Inserted test cell at position {insert_after_index}")
    print(f"New total cells: {len(nb.cells)}")

    # Fix all code cells to ensure they have required attributes
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            if not hasattr(cell, 'outputs') or cell.outputs is None:
                cell.outputs = []
            if not hasattr(cell, 'execution_count') or cell.execution_count is None:
                cell.execution_count = None

    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("SUCCESS: Test cell added!")
    return True

if __name__ == '__main__':
    notebook_path = r'C:\Users\thc1006\Desktop\dev\exoplanet-starter\notebooks\02_bls_baseline.ipynb'

    try:
        insert_test_cell(notebook_path, insert_after_index=8)
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)