"""
Model Card Generator for Exoplanet Detection Models
Following ML best practices for model documentation
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np


def create_model_card(
    model_name: str,
    model_version: str,
    training_date: str,
    metrics: Dict[str, float],
    features: List[str],
    calibration_method: str,
    hyperparameters: Dict[str, Any],
    dataset_info: Dict[str, Any],
    random_state: int = 42,
    additional_notes: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a comprehensive Model Card following best practices

    Args:
        model_name: Model identifier (e.g., 'XGBoost_Exoplanet_Detector')
        model_version: Version string (e.g., 'v1.0.0')
        training_date: ISO format date (e.g., '2025-09-30')
        metrics: Dict of evaluation metrics (pr_auc, roc_auc, brier_score, etc.)
        features: List of feature names used
        calibration_method: 'isotonic', 'sigmoid', or 'none'
        hyperparameters: XGBoost hyperparameters dict
        dataset_info: Dataset statistics (n_samples, n_positives, n_negatives)
        random_state: Random seed used
        additional_notes: Optional additional information

    Returns:
        Dict containing complete model card
    """
    model_card = {
        "model_details": {
            "name": model_name,
            "version": model_version,
            "date": training_date,
            "developer": "NASA Space Apps Exoplanet Team",
            "model_type": "XGBoost Gradient Boosting Classifier",
            "paper_or_resource": "https://github.com/exoplanet-spaceapps/exoplanet-starter"
        },
        "intended_use": {
            "primary_uses": "Exoplanet candidate detection and ranking",
            "primary_users": "Astronomers, citizen scientists, researchers",
            "out_of_scope": "Not for final confirmation; requires follow-up verification"
        },
        "training_data": {
            "dataset": dataset_info.get("name", "NASA TOI + KOI Combined"),
            "n_samples": dataset_info.get("n_samples", 0),
            "n_positive": dataset_info.get("n_positives", 0),
            "n_negative": dataset_info.get("n_negatives", 0),
            "class_balance": f"{dataset_info.get('n_positives', 0) / max(dataset_info.get('n_samples', 1), 1):.3f}",
            "features": features,
            "n_features": len(features),
            "data_source": "NASA TESS/Kepler via TAP/MAST queries"
        },
        "model_architecture": {
            "algorithm": "XGBoost",
            "preprocessing": [
                "SimpleImputer(strategy='median')",
                "RobustScaler()"
            ],
            "calibration": calibration_method,
            "hyperparameters": hyperparameters,
            "random_state": random_state
        },
        "performance": {
            "metrics": metrics,
            "evaluation_method": "StratifiedGroupKFold (5-fold)",
            "test_set_size": dataset_info.get("test_size", "20%")
        },
        "ethical_considerations": {
            "biases": [
                "May favor TESS/Kepler detection characteristics",
                "Class imbalance toward false positives"
            ],
            "limitations": [
                "Requires calibrated light curves",
                "Performance degrades on low SNR targets",
                "Not validated on other missions (JWST, etc.)"
            ],
            "recommendations": [
                "Always use probability calibration",
                "Verify high-confidence predictions with followup",
                "Monitor for distribution shift on new data"
            ]
        },
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "python_version": "3.10+",
            "key_dependencies": {
                "xgboost": ">=2.0.0",
                "scikit-learn": ">=1.3.0",
                "numpy": "1.26.4",
                "lightkurve": "latest"
            }
        }
    }

    if additional_notes:
        model_card["additional_notes"] = additional_notes

    return model_card


def save_model_card(model_card: Dict[str, Any], output_path: Path) -> None:
    """
    Save model card to JSON file

    Args:
        model_card: Model card dictionary
        output_path: Path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(model_card, f, indent=2, ensure_ascii=False)

    print(f"✅ Model Card saved to: {output_path}")


def load_model_card(path: Path) -> Dict[str, Any]:
    """
    Load model card from JSON file

    Args:
        path: Path to model card JSON

    Returns:
        Model card dictionary
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)