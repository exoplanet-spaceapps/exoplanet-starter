"""
合成凌日注入：在真實光曲線加入箱形下降
"""
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
import pandas as pd
from pathlib import Path
import json


def inject_box_transit(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    depth: float,
    duration: float,
    t0: float
) -> np.ndarray:
    """
    將箱形凌日注入 flux

    Parameters:
    -----------
    time : np.ndarray
        時間序列
    flux : np.ndarray
        原始流量
    period : float
        軌道週期（天）
    depth : float
        凌日深度（相對流量）
    duration : float
        凌日持續時間（天）
    t0 : float
        第一次凌日中心時刻

    Returns:
    --------
    np.ndarray : 注入凌日後的流量
    """
    model = flux.copy()

    # 計算相位（確保凌日在相位中心）
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0  # 中心化相位到 [-0.5, 0.5]

    # 判斷是否在凌日內
    half_duration_phase = (duration / period) / 2.0
    in_transit = np.abs(phase) < half_duration_phase

    # 注入凌日
    model[in_transit] *= (1.0 - depth)

    return model


def generate_synthetic_dataset(
    base_time: np.ndarray,
    base_flux: np.ndarray,
    n_positive: int = 200,
    n_negative: int = 200,
    period_range: Tuple[float, float] = (0.6, 10.0),
    depth_range: Tuple[float, float] = (0.0005, 0.02),
    duration_fraction_range: Tuple[float, float] = (0.02, 0.1),
    noise_level: float = 0.0001,
    seed: Optional[int] = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    生成合成訓練資料集

    Parameters:
    -----------
    base_time : np.ndarray
        基礎時間序列
    base_flux : np.ndarray
        基礎流量序列（已去趨勢）
    n_positive : int
        正樣本數量（有凌日）
    n_negative : int
        負樣本數量（無凌日）
    period_range : tuple
        週期範圍（天）
    depth_range : tuple
        深度範圍（相對流量）
    duration_fraction_range : tuple
        持續時間佔週期的比例範圍
    noise_level : float
        噪音水平
    seed : int
        隨機種子

    Returns:
    --------
    tuple : (光曲線資料, 標籤與參數資料)
    """
    if seed is not None:
        np.random.seed(seed)

    samples = []
    labels = []

    # 生成正樣本（有凌日）
    for i in range(n_positive):
        # 隨機生成凌日參數
        period = np.random.uniform(*period_range)
        depth = np.random.uniform(*depth_range)
        duration_fraction = np.random.uniform(*duration_fraction_range)
        duration = period * duration_fraction
        t0 = np.random.uniform(base_time[0], base_time[0] + period)

        # 注入凌日
        flux_with_transit = inject_box_transit(
            base_time, base_flux, period, depth, duration, t0
        )

        # 添加噪音
        noise = np.random.normal(0, noise_level, len(flux_with_transit))
        flux_with_transit += noise

        samples.append({
            'sample_id': f'positive_{i:04d}',
            'time': base_time.tolist(),
            'flux': flux_with_transit.tolist(),
            'label': 1,
            'true_period': period,
            'true_depth': depth,
            'true_duration': duration,
            'true_t0': t0,
            'noise_level': noise_level
        })

        labels.append({
            'sample_id': f'positive_{i:04d}',
            'label': 1,
            'has_transit': True,
            'period': period,
            'depth': depth,
            'duration': duration,
            't0': t0,
            'snr_estimate': depth / noise_level
        })

    # 生成負樣本（無凌日）
    for i in range(n_negative):
        # 只加噪音，不注入凌日
        flux_no_transit = base_flux.copy()

        # 添加隨機噪音和系統性變化
        noise = np.random.normal(0, noise_level, len(flux_no_transit))

        # 可選：添加一些非凌日的系統性變化
        if np.random.random() > 0.5:
            # 添加正弦變化（模擬恆星自轉）
            rotation_period = np.random.uniform(1, 20)
            rotation_amp = np.random.uniform(0.0001, 0.001)
            flux_no_transit += rotation_amp * np.sin(2 * np.pi * base_time / rotation_period)

        flux_no_transit += noise

        samples.append({
            'sample_id': f'negative_{i:04d}',
            'time': base_time.tolist(),
            'flux': flux_no_transit.tolist(),
            'label': 0,
            'true_period': None,
            'true_depth': None,
            'true_duration': None,
            'true_t0': None,
            'noise_level': noise_level
        })

        labels.append({
            'sample_id': f'negative_{i:04d}',
            'label': 0,
            'has_transit': False,
            'period': None,
            'depth': None,
            'duration': None,
            't0': None,
            'snr_estimate': None
        })

    # 轉換為 DataFrame
    samples_df = pd.DataFrame(samples)
    labels_df = pd.DataFrame(labels)

    return samples_df, labels_df


def save_synthetic_dataset(
    samples_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    output_dir: str = "data/synthetic",
    format: str = "parquet"
) -> Dict[str, str]:
    """
    儲存合成資料集

    Parameters:
    -----------
    samples_df : pd.DataFrame
        光曲線資料
    labels_df : pd.DataFrame
        標籤與參數資料
    output_dir : str
        輸出目錄
    format : str
        儲存格式 ('parquet' 或 'csv')

    Returns:
    --------
    dict : 儲存的檔案路徑
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {}

    if format == "parquet":
        samples_path = output_path / "synthetic_samples.parquet"
        labels_path = output_path / "synthetic_labels.parquet"
        samples_df.to_parquet(samples_path, index=False)
        labels_df.to_parquet(labels_path, index=False)
    else:  # csv
        samples_path = output_path / "synthetic_samples.csv"
        labels_path = output_path / "synthetic_labels.csv"
        samples_df.to_csv(samples_path, index=False)
        labels_df.to_csv(labels_path, index=False)

    paths['samples'] = str(samples_path)
    paths['labels'] = str(labels_path)

    # 儲存 metadata
    metadata = {
        'n_samples': len(samples_df),
        'n_positive': len(samples_df[samples_df['label'] == 1]),
        'n_negative': len(samples_df[samples_df['label'] == 0]),
        'format': format,
        'paths': paths
    }

    metadata_path = output_path / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    paths['metadata'] = str(metadata_path)

    return paths


def load_synthetic_dataset(
    dataset_dir: str = "data/synthetic"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    載入合成資料集

    Parameters:
    -----------
    dataset_dir : str
        資料集目錄

    Returns:
    --------
    tuple : (光曲線資料, 標籤與參數資料)
    """
    dataset_path = Path(dataset_dir)
    metadata_path = dataset_path / "dataset_metadata.json"

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    format = metadata['format']

    if format == "parquet":
        samples_df = pd.read_parquet(metadata['paths']['samples'])
        labels_df = pd.read_parquet(metadata['paths']['labels'])
    else:  # csv
        samples_df = pd.read_csv(metadata['paths']['samples'])
        labels_df = pd.read_csv(metadata['paths']['labels'])

    return samples_df, labels_df


def generate_transit_parameters(
    n_samples: int,
    period_range: Tuple[float, float] = (0.6, 10.0),
    depth_range: Tuple[float, float] = (0.0005, 0.02),
    duration_fraction_range: Tuple[float, float] = (0.02, 0.1),
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    生成凌日參數分布

    Parameters:
    -----------
    n_samples : int
        樣本數量
    period_range : tuple
        週期範圍（天）
    depth_range : tuple
        深度範圍（相對流量）
    duration_fraction_range : tuple
        持續時間佔週期的比例範圍
    seed : int
        隨機種子

    Returns:
    --------
    pd.DataFrame : 凌日參數表
    """
    if seed is not None:
        np.random.seed(seed)

    params = []

    for i in range(n_samples):
        period = np.random.uniform(*period_range)
        depth = np.random.uniform(*depth_range)
        duration_fraction = np.random.uniform(*duration_fraction_range)
        duration = period * duration_fraction
        t0 = np.random.uniform(0, period)

        # 計算額外的物理參數
        depth_ppm = depth * 1e6  # 轉換為 ppm
        duration_hours = duration * 24  # 轉換為小時

        params.append({
            'sample_id': f'sample_{i:04d}',
            'period_days': period,
            'depth_relative': depth,
            'depth_ppm': depth_ppm,
            'duration_days': duration,
            'duration_hours': duration_hours,
            'duration_fraction': duration_fraction,
            't0_days': t0,
            'impact_parameter': np.random.uniform(0, 0.9),  # 撞擊參數
            'stellar_radius': np.random.uniform(0.8, 1.2),  # 恆星半徑（太陽半徑）
            'planet_radius': np.sqrt(depth) * np.random.uniform(0.8, 1.2)  # 行星半徑估計
        })

    return pd.DataFrame(params)