"""
標準化候選輸出架構與驗證
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime


# 定義標準化欄位架構
CANDIDATE_SCHEMA = {
    # 基本識別欄位
    'target_id': str,           # TIC/KIC ID
    'mission': str,             # TESS/Kepler
    'sector_or_quarter': str,   # 扇區或季度編號

    # BLS 檢測參數
    'bls_period_d': float,      # 週期（天）
    'bls_duration_hr': float,   # 凌日持續時間（小時）
    'bls_depth_ppm': float,     # 凌日深度（ppm）
    'bls_t0': float,            # 第一次凌日時刻（BJD/BTJD）
    'snr': float,               # 信噪比
    'power': float,             # BLS 功率

    # 模型預測
    'model_score': float,       # 校準後機率
    'score_uncalibrated': float,# 原始分數

    # 質量標記
    'is_eb_flag': bool,         # 可能的食變星標記
    'toi_crossmatch': Optional[str],  # TOI 交叉比對結果
    'quality_flags': str,       # 質量標記（JSON 字串）

    # 元資料
    'run_id': str,              # 執行 ID
    'model_version': str,       # 模型版本
    'data_source_url': str,     # 資料來源 URL

    # NASA 欄位相容（可選）
    'pscomp_pl_rade': Optional[float],   # 行星半徑估計（地球半徑）
    'pscomp_pl_orbper': Optional[float], # 軌道週期（天）
    'pscomp_st_teff': Optional[float],   # 恆星有效溫度（K）
}


def create_candidate_dataframe(
    results: List[Dict[str, Any]],
    run_id: Optional[str] = None,
    model_version: str = "v1.0"
) -> pd.DataFrame:
    """
    將推論結果轉換為標準化候選資料框

    Parameters:
    -----------
    results : list
        推論結果列表（來自 predict_batch 等）
    run_id : str
        執行 ID（預設為時間戳記）
    model_version : str
        模型版本

    Returns:
    --------
    pd.DataFrame : 標準化候選資料框
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    candidates = []

    for result in results:
        if not result.get('success', False):
            continue

        # 提取基本資訊
        tic_id = result.get('tic_id', '')

        # 提取光曲線元資料
        lc_meta = result.get('lightcurve', {})
        mission = lc_meta.get('mission', 'TESS')
        sector = lc_meta.get('sector', 'unknown')

        # 提取 BLS 結果
        bls_period = result.get('bls_period', np.nan)
        bls_depth = result.get('bls_depth', np.nan)
        bls_snr = result.get('bls_snr', np.nan)

        # 從 features 提取更多資訊
        features = result.get('features', {})
        bls_duration = features.get('bls_duration', np.nan)
        bls_power = features.get('bls_power', np.nan)
        bls_t0 = features.get('bls_t0', np.nan)

        # 預測機率
        probability = result.get('probability', np.nan)

        # 質量標記
        is_eb = features.get('is_eb_flag', False)
        odd_even_diff = abs(features.get('odd_even_depth_diff', 0))
        transit_sym = features.get('transit_symmetry', 0)

        quality_flags = {
            'high_odd_even_diff': odd_even_diff > 0.01,
            'asymmetric_transit': abs(transit_sym) > 0.3,
            'low_snr': bls_snr < 7,
            'short_period': bls_period < 1.0
        }

        # 資料來源 URL
        data_source_url = f"https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery={tic_id}"

        # 建立候選記錄
        candidate = {
            'target_id': tic_id,
            'mission': mission,
            'sector_or_quarter': str(sector),
            'bls_period_d': float(bls_period) if not np.isnan(bls_period) else None,
            'bls_duration_hr': float(bls_duration * 24) if not np.isnan(bls_duration) else None,  # 轉為小時
            'bls_depth_ppm': float(bls_depth * 1e6) if not np.isnan(bls_depth) else None,  # 轉為 ppm
            'bls_t0': float(bls_t0) if not np.isnan(bls_t0) else None,
            'snr': float(bls_snr) if not np.isnan(bls_snr) else None,
            'power': float(bls_power) if not np.isnan(bls_power) else None,
            'model_score': float(probability) if not np.isnan(probability) else None,
            'score_uncalibrated': float(probability) if not np.isnan(probability) else None,  # 如有校準模型可區分
            'is_eb_flag': bool(is_eb),
            'toi_crossmatch': None,  # 可後續添加 TOI 比對功能
            'quality_flags': json.dumps(quality_flags),
            'run_id': run_id,
            'model_version': model_version,
            'data_source_url': data_source_url,
            'pscomp_pl_rade': None,  # 可選：估計行星半徑
            'pscomp_pl_orbper': float(bls_period) if not np.isnan(bls_period) else None,
            'pscomp_st_teff': None   # 可選：恆星溫度
        }

        candidates.append(candidate)

    # 轉為 DataFrame
    df = pd.DataFrame(candidates)

    # 按 model_score 降序排序
    if 'model_score' in df.columns:
        df = df.sort_values('model_score', ascending=False, na_position='last')

    return df


def validate_candidate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    驗證候選資料框是否符合標準架構

    Parameters:
    -----------
    df : pd.DataFrame
        候選資料框

    Returns:
    --------
    dict : 驗證結果
    """
    validation = {
        'valid': True,
        'missing_columns': [],
        'invalid_types': [],
        'warnings': []
    }

    # 檢查必要欄位
    required_columns = [
        'target_id', 'mission', 'bls_period_d', 'bls_depth_ppm',
        'snr', 'model_score', 'run_id', 'model_version'
    ]

    for col in required_columns:
        if col not in df.columns:
            validation['missing_columns'].append(col)
            validation['valid'] = False

    # 檢查資料型別（略過 None 值）
    if 'target_id' in df.columns and not df['target_id'].dtype == object:
        validation['invalid_types'].append('target_id should be string')

    if 'model_score' in df.columns:
        non_null_scores = df['model_score'].dropna()
        if len(non_null_scores) > 0:
            if not all((non_null_scores >= 0) & (non_null_scores <= 1)):
                validation['warnings'].append('model_score contains values outside [0, 1]')

    if 'snr' in df.columns:
        low_snr_count = (df['snr'] < 7).sum()
        if low_snr_count > 0:
            validation['warnings'].append(f'{low_snr_count} candidates have SNR < 7')

    return validation


def export_candidates_csv(
    df: pd.DataFrame,
    output_path: str = "outputs/candidates.csv",
    validate: bool = True
) -> str:
    """
    匯出候選清單為標準 CSV

    Parameters:
    -----------
    df : pd.DataFrame
        候選資料框
    output_path : str
        輸出路徑
    validate : bool
        是否驗證架構

    Returns:
    --------
    str : 輸出路徑
    """
    # 驗證架構
    if validate:
        validation = validate_candidate_schema(df)
        if not validation['valid']:
            raise ValueError(f"Schema validation failed: {validation['missing_columns']}")

        if validation['warnings']:
            print("⚠️ 警告:")
            for warning in validation['warnings']:
                print(f"   - {warning}")

    # 建立輸出目錄
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 匯出 CSV
    df.to_csv(output_path, index=False)

    print(f"✅ 匯出 {len(df)} 筆候選至: {output_path}")

    return str(output_path)


def export_candidates_jsonl(
    df: pd.DataFrame,
    output_path: str = "outputs/candidates.jsonl"
) -> str:
    """
    匯出候選清單為 JSONL 格式（每行一個 JSON 物件）

    Parameters:
    -----------
    df : pd.DataFrame
        候選資料框
    output_path : str
        輸出路徑

    Returns:
    --------
    str : 輸出路徑
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 轉為記錄列表
    records = df.to_dict(orient='records')

    # 寫入 JSONL
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            # 處理 NaN 值
            clean_record = {
                k: (None if pd.isna(v) else v)
                for k, v in record.items()
            }
            f.write(json.dumps(clean_record, ensure_ascii=False) + '\n')

    print(f"✅ 匯出 {len(df)} 筆候選至: {output_path}")

    return str(output_path)