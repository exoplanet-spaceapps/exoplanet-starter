"""
Provenance Tracking for Exoplanet Detection Pipeline
Tracks data sources, model versions, parameters, and timestamps
"""
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys
import importlib.metadata


def get_package_version(package_name: str) -> str:
    """Get installed package version"""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def create_provenance_record(
    run_id: str,
    data_source: str,
    mission: str,
    query_params: Dict[str, Any],
    model_info: Dict[str, Any],
    processing_steps: List[str],
    output_files: List[str],
    additional_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create comprehensive provenance record

    Args:
        run_id: Unique run identifier (e.g., '20250930_143022')
        data_source: Data source (e.g., 'MAST TAP', 'Local CSV')
        mission: Mission name (e.g., 'TESS', 'Kepler')
        query_params: TAP query parameters or file paths
        model_info: Model version, path, hyperparameters
        processing_steps: List of processing steps performed
        output_files: List of output file paths
        additional_metadata: Optional additional metadata

    Returns:
        Provenance record dictionary
    """
    provenance = {
        "run_info": {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform
        },
        "data_source": {
            "source": data_source,
            "mission": mission,
            "query_parameters": query_params,
            "query_timestamp": datetime.now().isoformat()
        },
        "model": {
            "model_version": model_info.get("version", "unknown"),
            "model_path": model_info.get("path", ""),
            "calibration_method": model_info.get("calibration", "none"),
            "hyperparameters": model_info.get("hyperparameters", {}),
            "random_state": model_info.get("random_state", 42)
        },
        "dependencies": {
            "lightkurve": get_package_version("lightkurve"),
            "numpy": get_package_version("numpy"),
            "pandas": get_package_version("pandas"),
            "xgboost": get_package_version("xgboost"),
            "scikit-learn": get_package_version("scikit-learn"),
            "astropy": get_package_version("astropy"),
            "wotan": get_package_version("wotan")
        },
        "processing": {
            "steps": processing_steps,
            "start_time": datetime.now().isoformat()
        },
        "outputs": {
            "files": output_files,
            "output_directory": str(Path(output_files[0]).parent) if output_files else ""
        }
    }

    if additional_metadata:
        provenance["additional_metadata"] = additional_metadata

    return provenance


def save_provenance(provenance: Dict[str, Any], output_path: Path) -> None:
    """
    Save provenance record to YAML file

    Args:
        provenance: Provenance record dictionary
        output_path: Path to save YAML file (e.g., outputs/provenance.yaml)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(provenance, f, default_flow_style=False, sort_keys=False)

    print(f"✅ Provenance saved to: {output_path}")


def load_provenance(path: Path) -> Dict[str, Any]:
    """
    Load provenance record from YAML file

    Args:
        path: Path to provenance YAML file

    Returns:
        Provenance dictionary
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def update_provenance(path: Path, updates: Dict[str, Any]) -> None:
    """
    Update existing provenance file with new information

    Args:
        path: Path to existing provenance file
        updates: Dictionary of updates to merge
    """
    provenance = load_provenance(path)

    # Deep merge updates
    def deep_merge(base: dict, updates: dict) -> dict:
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
        return base

    provenance = deep_merge(provenance, updates)
    provenance["processing"]["last_updated"] = datetime.now().isoformat()

    save_provenance(provenance, path)