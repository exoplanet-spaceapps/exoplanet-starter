"""
資料來源追蹤與執行元資料記錄
"""
from typing import Dict, Optional, Any, List
import yaml
import json
from pathlib import Path
from datetime import datetime
import platform
import sys


def create_provenance_record(
    run_id: str,
    query_params: Optional[Dict[str, Any]] = None,
    model_info: Optional[Dict[str, Any]] = None,
    package_versions: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    建立完整的資料來源追蹤記錄

    Parameters:
    -----------
    run_id : str
        執行 ID
    query_params : dict
        查詢參數（例如 TIC 列表、任務名稱等）
    model_info : dict
        模型資訊（版本、路徑等）
    package_versions : dict
        套件版本資訊

    Returns:
    --------
    dict : 資料來源記錄
    """
    # 自動偵測套件版本
    if package_versions is None:
        package_versions = {}

        try:
            import lightkurve
            package_versions['lightkurve'] = lightkurve.__version__
        except:
            package_versions['lightkurve'] = 'not installed'

        try:
            import numpy
            package_versions['numpy'] = numpy.__version__
        except:
            pass

        try:
            import pandas
            package_versions['pandas'] = pandas.__version__
        except:
            pass

        try:
            import sklearn
            package_versions['scikit-learn'] = sklearn.__version__
        except:
            pass

        try:
            import xgboost
            package_versions['xgboost'] = xgboost.__version__
        except:
            pass

    # 建立記錄
    provenance = {
        'run_info': {
            'run_id': run_id,
            'timestamp': datetime.now().isoformat(),
            'platform': platform.platform(),
            'python_version': sys.version,
            'hostname': platform.node()
        },

        'query_parameters': query_params or {},

        'model_information': model_info or {
            'version': 'unknown',
            'type': 'unknown',
            'path': 'unknown'
        },

        'software_versions': package_versions,

        'data_sources': {
            'lightcurve_archive': 'MAST (https://mast.stsci.edu)',
            'catalog': 'NASA Exoplanet Archive',
            'method': 'Lightkurve API',
            'detrending': 'Savitzky-Golay (window=401)'
        },

        'processing_steps': [
            {
                'step': 1,
                'name': 'Data Download',
                'description': 'Download lightcurves from MAST via Lightkurve'
            },
            {
                'step': 2,
                'name': 'Preprocessing',
                'description': 'Remove NaNs, flatten lightcurve with Savitzky-Golay filter'
            },
            {
                'step': 3,
                'name': 'BLS Search',
                'description': 'Run Box Least Squares periodogram search'
            },
            {
                'step': 4,
                'name': 'Feature Extraction',
                'description': 'Extract transit and stellar features'
            },
            {
                'step': 5,
                'name': 'Model Prediction',
                'description': 'Predict planet candidate probability using trained model'
            }
        ],

        'quality_control': {
            'min_snr_threshold': 7.0,
            'min_period_days': 0.5,
            'max_period_days': 20.0,
            'min_datapoints': 100
        }
    }

    return provenance


def save_provenance(
    provenance: Dict[str, Any],
    output_path: str = "outputs/provenance.yaml"
) -> str:
    """
    儲存資料來源記錄為 YAML 檔案

    Parameters:
    -----------
    provenance : dict
        資料來源記錄
    output_path : str
        輸出路徑

    Returns:
    --------
    str : 輸出路徑
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 儲存為 YAML
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(provenance, f, default_flow_style=False, allow_unicode=True)

    print(f"✅ 資料來源記錄已儲存: {output_path}")

    return str(output_path)


def load_provenance(file_path: str) -> Dict[str, Any]:
    """
    載入資料來源記錄

    Parameters:
    -----------
    file_path : str
        檔案路徑

    Returns:
    --------
    dict : 資料來源記錄
    """
    file_path = Path(file_path)

    if file_path.suffix == '.yaml' or file_path.suffix == '.yml':
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def add_execution_metadata(
    provenance: Dict[str, Any],
    n_targets: int,
    n_success: int,
    n_high_confidence: int,
    execution_time_seconds: Optional[float] = None
) -> Dict[str, Any]:
    """
    添加執行結果元資料到資料來源記錄

    Parameters:
    -----------
    provenance : dict
        現有資料來源記錄
    n_targets : int
        處理目標數
    n_success : int
        成功處理數
    n_high_confidence : int
        高信心候選數
    execution_time_seconds : float
        執行時間（秒）

    Returns:
    --------
    dict : 更新後的資料來源記錄
    """
    provenance['execution_results'] = {
        'total_targets': n_targets,
        'successful_predictions': n_success,
        'high_confidence_candidates': n_high_confidence,
        'success_rate': f"{n_success/n_targets*100:.1f}%" if n_targets > 0 else "N/A"
    }

    if execution_time_seconds is not None:
        provenance['execution_results']['execution_time_seconds'] = execution_time_seconds
        provenance['execution_results']['avg_time_per_target'] = f"{execution_time_seconds/n_targets:.2f}s" if n_targets > 0 else "N/A"

    return provenance