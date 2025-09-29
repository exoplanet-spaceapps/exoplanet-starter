"""
Standardized Output Schema for Exoplanet Candidate Export
Following best practices for reproducible data science
"""
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime


def create_candidate_dataframe(
    target_ids: List[str],
    missions: List[str],
    bls_results: Dict[str, Any],
    model_predictions: Dict[str, float],
    additional_features: Optional[pd.DataFrame] = None,
    run_id: Optional[str] = None,
    model_version: str = "v1.0.0"
) -> pd.DataFrame:
    """
    Create standardized candidate output DataFrame

    Args:
        target_ids: List of target identifiers (e.g., ['TIC123456', ...])
        missions: List of mission names (e.g., ['TESS', 'Kepler', ...])
        bls_results: Dict containing BLS analysis results
        model_predictions: Dict mapping target_id to model scores
        additional_features: Optional DataFrame with extra features
        run_id: Run identifier (defaults to timestamp)
        model_version: Model version string

    Returns:
        DataFrame with standardized schema
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Base schema
    df = pd.DataFrame({
        # Identifiers
        "target_id": target_ids,
        "mission": missions,
        "sector_or_quarter": [bls_results.get(tid, {}).get("sector", "") for tid in target_ids],

        # BLS Features
        "bls_period_d": [bls_results.get(tid, {}).get("period", np.nan) for tid in target_ids],
        "bls_duration_hr": [bls_results.get(tid, {}).get("duration", np.nan) for tid in target_ids],
        "bls_depth_ppm": [bls_results.get(tid, {}).get("depth", np.nan) for tid in target_ids],
        "snr": [bls_results.get(tid, {}).get("snr", np.nan) for tid in target_ids],
        "power": [bls_results.get(tid, {}).get("power", np.nan) for tid in target_ids],

        # Model Predictions
        "model_score": [model_predictions.get(tid, np.nan) for tid in target_ids],
        "score_uncalibrated": [model_predictions.get(f"{tid}_uncalib", np.nan) for tid in target_ids],

        # Quality Flags
        "is_eb_flag": [bls_results.get(tid, {}).get("is_eb", False) for tid in target_ids],
        "toi_crossmatch": [bls_results.get(tid, {}).get("toi_match", "") for tid in target_ids],
        "quality_flags": [bls_results.get(tid, {}).get("flags", "") for tid in target_ids],

        # Metadata
        "run_id": run_id,
        "model_version": model_version,
        "data_source_url": [bls_results.get(tid, {}).get("url", "") for tid in target_ids]
    })

    # Add planetary comparison features if available
    if additional_features is not None and 'pscomp_pl_rade' in additional_features.columns:
        df = df.merge(
            additional_features[['target_id', 'pscomp_pl_rade', 'pscomp_pl_orbper', 'pscomp_st_teff']],
            on='target_id',
            how='left'
        )
    else:
        df['pscomp_pl_rade'] = np.nan
        df['pscomp_pl_orbper'] = np.nan
        df['pscomp_st_teff'] = np.nan

    # Sort by model score (descending)
    df = df.sort_values('model_score', ascending=False).reset_index(drop=True)

    return df


def export_candidates_csv(
    df: pd.DataFrame,
    output_dir: Path = Path("outputs"),
    timestamp: Optional[str] = None
) -> Path:
    """
    Export candidates to CSV with timestamp

    Args:
        df: Candidate DataFrame with standardized schema
        output_dir: Output directory (default: outputs/)
        timestamp: Optional timestamp string (defaults to current time)

    Returns:
        Path to exported CSV file
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"candidates_{timestamp}.csv"

    df.to_csv(output_path, index=False, float_format='%.6f')

    print(f"✅ Candidates exported to: {output_path}")
    print(f"   Total candidates: {len(df)}")
    print(f"   High confidence (score > 0.8): {(df['model_score'] > 0.8).sum()}")

    return output_path


def export_candidates_jsonl(
    df: pd.DataFrame,
    output_dir: Path = Path("outputs"),
    timestamp: Optional[str] = None
) -> Path:
    """
    Export candidates to JSONL format (one JSON per line)

    Args:
        df: Candidate DataFrame
        output_dir: Output directory
        timestamp: Optional timestamp string

    Returns:
        Path to exported JSONL file
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"candidates_{timestamp}.jsonl"

    df.to_json(output_path, orient='records', lines=True)

    print(f"✅ Candidates exported to: {output_path} (JSONL)")

    return output_path


import numpy as np  # Add import at top for np.nan usage