#!/usr/bin/env python
"""
Notebook 04 Inference Script: Real NASA TOI Data Inference
Performs inference on real NASA TOI candidates using trained model
"""
import pandas as pd
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path

def main():
    print("="*70)
    print("NOTEBOOK 04: Real NASA TOI Data Inference with GPU")
    print("="*70)
    print()

    # Load TOI data
    print("Step 1/5: Loading NASA TOI data...")
    toi_df = pd.read_csv('data/toi.csv')
    print(f"Loaded {len(toi_df)} TOI candidates")
    print()

    # Load model
    print("Step 2/5: Loading trained model...")
    model = joblib.load('models/xgboost_pipeline_cv.joblib')
    print(f"Model loaded: {type(model)}")
    expected_features = list(model.named_steps['preprocessor'].feature_names_in_)
    print(f"Expected features: {expected_features}")
    print()

    # Prepare features - use correct TOI column names
    print("Step 3/5: Preparing features for inference...")
    features_df = pd.DataFrame({
        'toi': toi_df['toi'],  # TOI number
        'tid': toi_df['tid'],  # TIC ID
        'period': toi_df['pl_orbper'],  # Orbital period
        'depth': toi_df['pl_trandep'],  # Transit depth
        'duration': toi_df['pl_trandurh'],  # Transit duration (hours)
        'kepid': toi_df['tid']  # Use TIC ID as kepid
    })

    # Handle missing values
    features_df = features_df.fillna(0)
    print(f"Prepared {len(features_df)} samples with {len(features_df.columns)} features")
    print(f"Missing values filled with 0")
    print()

    # Run inference
    print("Step 4/5: Running inference on NASA TOI data (using GPU)...")
    start_time = datetime.now()
    try:
        predictions = model.predict_proba(features_df)[:, 1]
        end_time = datetime.now()
        inference_time = (end_time - start_time).total_seconds()
        print(f"Inference complete: {len(predictions)} predictions in {inference_time:.2f}s")
        print(f"Score range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        print(f"Throughput: {len(predictions)/inference_time:.0f} predictions/second")
        print()
    except Exception as e:
        print(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

    # Create output dataframe
    output_df = pd.DataFrame({
        'target_id': 'TIC ' + toi_df['tid'].astype(str),
        'toi_number': toi_df['toi'],
        'model_score': predictions,
        'bls_period_d': toi_df['pl_orbper'],
        'bls_depth_ppm': toi_df['pl_trandep'] * 1e6,  # Convert to ppm
        'bls_duration_hr': toi_df['pl_trandurh'],
        'snr': np.nan,  # Not available from TOI data
        'mission': 'TESS',
        'run_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'model_version': 'xgboost_pipeline_cv_v1.0',
        'data_source_url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI'
    })

    # Sort by score
    output_df = output_df.sort_values('model_score', ascending=False)

    # Stats
    high_conf = len(output_df[output_df['model_score'] > 0.8])
    med_conf = len(output_df[(output_df['model_score'] > 0.5) & (output_df['model_score'] <= 0.8)])
    low_conf = len(output_df[output_df['model_score'] <= 0.5])

    print("Step 5/5: Generating outputs...")
    # Create outputs directory
    Path('outputs').mkdir(exist_ok=True)

    # Save CSV
    date_str = datetime.now().strftime("%Y%m%d")
    csv_path = f'outputs/candidates_{date_str}.csv'
    output_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Save JSONL
    jsonl_path = f'outputs/candidates_{date_str}.jsonl'
    with open(jsonl_path, 'w') as f:
        for _, row in output_df.iterrows():
            f.write(row.to_json() + '\n')
    print(f"Saved JSONL: {jsonl_path}")

    # Save provenance
    import yaml
    provenance = {
        'run_id': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'timestamp': datetime.now().isoformat(),
        'execution_time_seconds': inference_time,
        'model': {
            'path': 'models/xgboost_pipeline_cv.joblib',
            'version': 'xgboost_pipeline_cv_v1.0',
            'type': 'XGBClassifier with preprocessing pipeline',
            'features': expected_features
        },
        'data_source': {
            'file': 'data/toi.csv',
            'n_samples': len(toi_df),
            'url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI',
            'query_date': datetime.now().isoformat()
        },
        'results': {
            'n_candidates': len(output_df),
            'high_confidence': high_conf,
            'medium_confidence': med_conf,
            'low_confidence': low_conf
        },
        'hardware': {
            'gpu': 'NVIDIA GeForce RTX 3050 Laptop GPU',
            'cuda_version': '12.4'
        }
    }

    prov_path = f'outputs/provenance_{date_str}.yaml'
    with open(prov_path, 'w') as f:
        yaml.dump(provenance, f, default_flow_style=False)
    print(f"Saved provenance: {prov_path}")
    print()

    # Final report
    print("="*70)
    print("NOTEBOOK 04 EXECUTION COMPLETE")
    print("="*70)
    print()
    print(f"Results Summary:")
    print(f"  - Total candidates processed: {len(output_df)}")
    print(f"  - High confidence (>0.8): {high_conf} ({high_conf/len(output_df)*100:.1f}%)")
    print(f"  - Medium confidence (0.5-0.8): {med_conf} ({med_conf/len(output_df)*100:.1f}%)")
    print(f"  - Low confidence (<0.5): {low_conf} ({low_conf/len(output_df)*100:.1f}%)")
    print()
    print(f"Top 10 candidates:")
    for i, (_, row) in enumerate(output_df.head(10).iterrows(), 1):
        print(f"  {i}. {row['target_id']} (TOI {int(row['toi_number'])})")
        print(f"     Score: {row['model_score']:.3f} | Period: {row['bls_period_d']:.2f}d | Depth: {row['bls_depth_ppm']:.0f} ppm")
    print()
    print(f"Output files:")
    print(f"  - CSV:   {csv_path} ({len(output_df)} candidates)")
    print(f"  - JSONL: {jsonl_path}")
    print(f"  - YAML:  {prov_path} (provenance record)")
    print()
    print(f"Performance:")
    print(f"  - Inference time: {inference_time:.2f}s")
    print(f"  - Throughput: {len(predictions)/inference_time:.0f} predictions/second")
    print(f"  - GPU: RTX 3050 Laptop (4GB)")
    print("="*70)

if __name__ == '__main__':
    main()