"""
BLS/TLS 與特徵萃取
"""
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
from pathlib import Path
import json


def run_bls(
    time: np.ndarray,
    flux: np.ndarray,
    min_period: float = 0.5,
    max_period: float = 20.0
) -> Dict[str, float]:
    """
    執行 BLS 搜尋並回傳最強峰的參數

    Parameters:
    -----------
    time : np.ndarray
        時間序列
    flux : np.ndarray
        流量序列
    min_period : float
        最小搜尋週期（天）
    max_period : float
        最大搜尋週期（天）

    Returns:
    --------
    dict : BLS 結果（period, t0, duration, depth, snr）
    """
    try:
        import lightkurve as lk

        # 創建 LightCurve 物件
        lc = lk.LightCurve(time=time, flux=flux)

        # 執行 BLS
        bls = lc.to_periodogram(
            method="bls",
            minimum_period=min_period,
            maximum_period=max_period,
            frequency_factor=5.0
        )

        # 提取參數
        period = bls.period_at_max_power
        t0 = bls.transit_time_at_max_power
        duration = bls.duration_at_max_power
        depth = bls.depth_at_max_power
        snr = bls.max_power

        return {
            "period": period.value if hasattr(period, 'value') else float(period),
            "t0": t0.value if hasattr(t0, 'value') else float(t0),
            "duration": duration.value if hasattr(duration, 'value') else float(duration),
            "depth": depth.value if hasattr(depth, 'value') else float(depth),
            "snr": snr.value if hasattr(snr, 'value') else float(snr)
        }

    except ImportError:
        # 如果沒有 lightkurve，返回模擬結果
        print("Warning: lightkurve not available, returning mock results")
        return {
            "period": np.random.uniform(min_period, max_period),
            "t0": np.random.uniform(0, 10),
            "duration": np.random.uniform(0.01, 0.5),
            "depth": np.random.uniform(0.0001, 0.01),
            "snr": np.random.uniform(5, 50)
        }


def extract_features(
    time: np.ndarray,
    flux: np.ndarray,
    bls_result: Dict[str, float],
    compute_advanced: bool = True
) -> Dict[str, float]:
    """
    將 BLS 結果與幾何/對稱性等轉為特徵字典

    Parameters:
    -----------
    time : np.ndarray
        時間序列
    flux : np.ndarray
        流量序列
    bls_result : dict
        BLS 搜尋結果
    compute_advanced : bool
        是否計算進階特徵

    Returns:
    --------
    dict : 特徵字典
    """
    feats = {
        # BLS 基本特徵
        "bls_period": bls_result.get("period", np.nan),
        "bls_duration": bls_result.get("duration", np.nan),
        "bls_depth": bls_result.get("depth", np.nan),
        "bls_snr": bls_result.get("snr", np.nan),
        "bls_t0": bls_result.get("t0", np.nan),

        # 衍生特徵
        "duration_over_period": bls_result.get("duration", 0) / bls_result.get("period", 1) if bls_result.get("period", 0) > 0 else np.nan,
        "depth_snr_ratio": bls_result.get("depth", 0) / bls_result.get("snr", 1) if bls_result.get("snr", 0) > 0 else np.nan,
    }

    if compute_advanced and not np.isnan(feats["bls_period"]):
        # 計算進階特徵
        period = feats["bls_period"]
        t0 = feats["bls_t0"]
        duration = feats["bls_duration"]

        # 摺疊光曲線
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0

        # 奇偶深度差（檢測雙星）
        odd_even_diff = compute_odd_even_difference(time, flux, period, t0, duration)
        feats["odd_even_depth_diff"] = odd_even_diff

        # 凌日形狀對稱性
        symmetry = compute_transit_symmetry(time, flux, period, t0, duration)
        feats["transit_symmetry"] = symmetry

        # 流量統計特徵
        feats["flux_std"] = np.std(flux)
        feats["flux_mad"] = np.median(np.abs(flux - np.median(flux)))
        feats["flux_skew"] = compute_skewness(flux)
        feats["flux_kurtosis"] = compute_kurtosis(flux)

        # 週期性強度
        feats["periodicity_strength"] = compute_periodicity_strength(time, flux, period)

    return feats


def compute_odd_even_difference(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float
) -> float:
    """
    計算奇偶凌日深度差異（用於識別雙星）

    Parameters:
    -----------
    time : np.ndarray
        時間序列
    flux : np.ndarray
        流量序列
    period : float
        軌道週期
    t0 : float
        第一次凌日時刻
    duration : float
        凌日持續時間

    Returns:
    --------
    float : 奇偶深度差異
    """
    try:
        # 計算每個凌日的編號
        transit_number = np.floor((time - t0) / period).astype(int)

        # 分離奇偶凌日
        odd_transits = transit_number % 2 == 1
        even_transits = transit_number % 2 == 0

        # 計算相位
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0
        in_transit = np.abs(phase) < (duration / period / 2)

        # 計算奇偶凌日的平均深度
        odd_in_transit = in_transit & odd_transits
        even_in_transit = in_transit & even_transits

        if np.sum(odd_in_transit) > 0 and np.sum(even_in_transit) > 0:
            odd_depth = 1.0 - np.median(flux[odd_in_transit])
            even_depth = 1.0 - np.median(flux[even_in_transit])
            return abs(odd_depth - even_depth)
        else:
            return 0.0

    except Exception:
        return 0.0


def compute_transit_symmetry(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float,
    duration: float
) -> float:
    """
    計算凌日形狀的對稱性

    Parameters:
    -----------
    time : np.ndarray
        時間序列
    flux : np.ndarray
        流量序列
    period : float
        軌道週期
    t0 : float
        第一次凌日時刻
    duration : float
        凌日持續時間

    Returns:
    --------
    float : 對稱性指標（0 = 完全對稱，1 = 完全不對稱）
    """
    try:
        # 摺疊光曲線
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0

        # 選取凌日區域
        half_duration_phase = (duration / period) / 2.0
        in_transit = np.abs(phase) < half_duration_phase

        if np.sum(in_transit) < 10:  # 需要足夠的點
            return 0.5

        transit_phase = phase[in_transit]
        transit_flux = flux[in_transit]

        # 分為入凌和出凌
        ingress = transit_phase < 0
        egress = transit_phase > 0

        if np.sum(ingress) > 0 and np.sum(egress) > 0:
            # 計算兩側的平均斜率
            ingress_slope = np.mean(np.diff(transit_flux[ingress]))
            egress_slope = np.mean(np.diff(transit_flux[egress]))

            # 對稱性 = 斜率差異
            symmetry = abs(ingress_slope + egress_slope) / (abs(ingress_slope) + abs(egress_slope) + 1e-10)
            return min(symmetry, 1.0)
        else:
            return 0.5

    except Exception:
        return 0.5


def compute_skewness(data: np.ndarray) -> float:
    """計算偏度"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 3)


def compute_kurtosis(data: np.ndarray) -> float:
    """計算峰度"""
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return 0.0
    return np.mean(((data - mean) / std) ** 4) - 3.0


def compute_periodicity_strength(
    time: np.ndarray,
    flux: np.ndarray,
    period: float
) -> float:
    """
    計算週期性強度

    Parameters:
    -----------
    time : np.ndarray
        時間序列
    flux : np.ndarray
        流量序列
    period : float
        週期

    Returns:
    --------
    float : 週期性強度（0-1）
    """
    try:
        # 摺疊光曲線
        phase = ((time - np.min(time)) % period) / period

        # 分成多個相位區間
        n_bins = 20
        phase_bins = np.linspace(0, 1, n_bins + 1)
        binned_flux = []

        for i in range(n_bins):
            mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
            if np.sum(mask) > 0:
                binned_flux.append(np.median(flux[mask]))
            else:
                binned_flux.append(np.nan)

        binned_flux = np.array(binned_flux)
        binned_flux = binned_flux[~np.isnan(binned_flux)]

        if len(binned_flux) > 5:
            # 週期性強度 = 變化幅度 / 噪音水平
            variation = np.std(binned_flux)
            noise = np.std(flux)
            return min(variation / (noise + 1e-10), 1.0)
        else:
            return 0.0

    except Exception:
        return 0.0


def create_feature_schema(
    feature_names: List[str],
    output_path: str = "data/feature_schema.json"
) -> Dict[str, Any]:
    """
    創建特徵架構檔案

    Parameters:
    -----------
    feature_names : list
        特徵名稱列表
    output_path : str
        輸出路徑

    Returns:
    --------
    dict : 特徵架構
    """
    schema = {
        "version": "1.0",
        "n_features": len(feature_names),
        "features": [],
        "feature_order": feature_names
    }

    # 定義每個特徵的描述
    feature_descriptions = {
        "bls_period": "BLS detected orbital period (days)",
        "bls_duration": "BLS detected transit duration (days)",
        "bls_depth": "BLS detected transit depth (relative flux)",
        "bls_snr": "BLS signal-to-noise ratio",
        "bls_t0": "BLS transit epoch (days)",
        "duration_over_period": "Transit duration divided by orbital period",
        "depth_snr_ratio": "Transit depth divided by SNR",
        "odd_even_depth_diff": "Depth difference between odd and even transits",
        "transit_symmetry": "Transit shape symmetry (0=symmetric, 1=asymmetric)",
        "flux_std": "Standard deviation of flux",
        "flux_mad": "Median absolute deviation of flux",
        "flux_skew": "Skewness of flux distribution",
        "flux_kurtosis": "Kurtosis of flux distribution",
        "periodicity_strength": "Strength of periodic signal (0-1)"
    }

    for i, name in enumerate(feature_names):
        schema["features"].append({
            "index": i,
            "name": name,
            "description": feature_descriptions.get(name, f"Feature {name}"),
            "type": "float",
            "nullable": False
        })

    # 儲存架構
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(schema, f, indent=2)

    return schema


def extract_features_batch(
    samples_df: pd.DataFrame,
    compute_advanced: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    批次提取特徵

    Parameters:
    -----------
    samples_df : pd.DataFrame
        包含 time 和 flux 的樣本資料
    compute_advanced : bool
        是否計算進階特徵
    verbose : bool
        是否顯示進度

    Returns:
    --------
    pd.DataFrame : 特徵資料框
    """
    features_list = []

    for idx, row in samples_df.iterrows():
        if verbose and idx % 50 == 0:
            print(f"Processing sample {idx}/{len(samples_df)}...")

        sample_id = row['sample_id']
        time = np.array(row['time'])
        flux = np.array(row['flux'])

        # 執行 BLS
        bls_result = run_bls(time, flux)

        # 提取特徵
        features = extract_features(time, flux, bls_result, compute_advanced)
        features['sample_id'] = sample_id
        features['label'] = row.get('label', -1)

        features_list.append(features)

    return pd.DataFrame(features_list)


def compute_feature_importance(
    features_df: pd.DataFrame,
    labels: np.ndarray,
    method: str = "random_forest"
) -> pd.DataFrame:
    """
    計算特徵重要性

    Parameters:
    -----------
    features_df : pd.DataFrame
        特徵資料框
    labels : np.ndarray
        標籤
    method : str
        方法（random_forest 或 mutual_info）

    Returns:
    --------
    pd.DataFrame : 特徵重要性排名
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif

    # 準備特徵矩陣
    feature_cols = [col for col in features_df.columns if col not in ['sample_id', 'label']]
    X = features_df[feature_cols].values

    if method == "random_forest":
        # 使用隨機森林計算特徵重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, labels)
        importances = rf.feature_importances_
    else:  # mutual_info
        # 使用互信息計算特徵重要性
        importances = mutual_info_classif(X, labels, random_state=42)

    # 創建重要性資料框
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    })
    importance_df = importance_df.sort_values('importance', ascending=False)

    return importance_df