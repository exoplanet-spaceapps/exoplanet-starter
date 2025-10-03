"""
Manual execution of notebook 04 key cells
"""
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Setup paths
project_root = Path(__file__).parent.parent
os.chdir(project_root)  # Change to project root first

# Clean up sys.path and insert project root at the very beginning
sys.path = [p for p in sys.path if 'cyberbully' not in p]  # Remove any interfering paths
sys.path.insert(0, str(project_root))  # Insert project root FIRST

print("=== Notebook 04: New Data Inference Pipeline ===")
print(f"Project root: {project_root}")
print(f"Working dir: {os.getcwd()}")
print()

# Import required modules
print("Step 1: Importing modules...")
print(f"DEBUG - sys.path[0]: {sys.path[0]}")
print(f"DEBUG - os.getcwd(): {os.getcwd()}")
print(f"DEBUG - app dir exists: {os.path.exists('app')}")
print(f"DEBUG - app/__init__.py exists: {os.path.exists('app/__init__.py')}")
print(f"DEBUG - app/infer.py exists: {os.path.exists('app/infer.py')}")
try:
    import numpy as np
    import pandas as pd
    import joblib
    print("  - Basic imports OK")
    from app.infer import (
        predict_from_tic,
        predict_batch,
        create_folded_lightcurve_plot,
        save_inference_results,
        check_gpu_availability
    )
    from app.bls_features import run_bls, extract_features
    print("OK - Modules imported successfully")
except ImportError as e:
    print(f"ERROR - Import failed: {e}")
    sys.exit(1)

# Check GPU
print("\nStep 2: Checking GPU availability...")
gpu_info = check_gpu_availability()
print(f"GPU Available: {gpu_info.get('available', False)}")
if gpu_info.get('available'):
    print(f"Device: {gpu_info.get('device_name', 'Unknown')}")

# Load model
print("\nStep 3: Loading model...")
model_dir = project_root / "models"
model_path = model_dir / "xgboost_pipeline_cv.joblib"

if not model_path.exists():
    print(f"ERROR - Model not found: {model_path}")
    sys.exit(1)

try:
    model = joblib.load(model_path)
    print(f"OK - Model loaded: {model_path}")
except Exception as e:
    print(f"ERROR - Failed to load model: {e}")
    sys.exit(1)

# Single target inference
print("\nStep 4: Single target inference (TIC 25155310)...")
tic_id = "TIC 25155310"

try:
    result = predict_from_tic(
        tic_id,
        model_path=str(model_path),
        scaler_path=None,
        feature_schema_path=None,
        mission="TESS",
        verbose=False
    )

    if result['success']:
        print(f"OK - Inference successful")
        print(f"  Target: {result['tic_id']}")
        print(f"  Probability: {result['probability']:.3f}")
        print(f"  BLS Period: {result['bls_period']:.3f} days")
        print(f"  BLS SNR: {result['bls_snr']:.1f}")
    else:
        print(f"WARN - Inference failed: {result.get('error', 'Unknown')}")

except Exception as e:
    print(f"ERROR - Inference failed: {e}")
    import traceback
    traceback.print_exc()

# Batch inference
print("\nStep 5: Batch inference (5 targets)...")
tic_list = [
    "TIC 25155310",
    "TIC 307210830",
    "TIC 260004324",
    "TIC 55652896",
    "TIC 441462736",
]

try:
    results_df = predict_batch(
        tic_list,
        model_path=str(model_path),
        scaler_path=None,
        feature_schema_path=None,
        mission="TESS",
        verbose=False
    )

    print(f"OK - Batch inference complete")
    print(f"  Total targets: {len(results_df)}")
    print(f"  Successful: {results_df['success'].sum()}")

    # Show results
    if len(results_df) > 0:
        print("\n  Results:")
        for _, row in results_df.iterrows():
            if row['success']:
                print(f"    {row['tic_id']}: prob={row['probability']:.3f}, "
                      f"period={row['bls_period']:.2f}d, snr={row['bls_snr']:.1f}")
            else:
                print(f"    {row['tic_id']}: FAILED ({row.get('error', 'Unknown')})")

        # Save results
        output_dir = project_root / 'results'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'batch_inference.csv'

        results_df.to_csv(output_path, index=False)
        print(f"\n  Results saved: {output_path}")

except Exception as e:
    print(f"ERROR - Batch inference failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Execution Complete ===")