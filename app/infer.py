"""
推論管線：TIC -> MAST 下載 -> 特徵萃取 -> 預測
"""
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import lightkurve as lk
except ImportError:
    print("Warning: lightkurve not installed")
    lk = None

from .bls_features import run_bls, extract_features
# Note: Light curve download is handled directly by lightkurve in predict_from_tic()


def predict_from_tic(
    tic_id: str,
    model_path: str = "model/ranker.joblib",
    scaler_path: str = "model/scaler.joblib",
    feature_schema_path: str = "model/feature_schema.json",
    mission: str = "TESS",
    detrend_window: int = 401,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    從 TIC ID 進行端到端預測

    Parameters:
    -----------
    tic_id : str
        目標識別碼（例如 "TIC 25155310" 或 "25155310"）
    model_path : str
        訓練好的模型路徑
    scaler_path : str
        特徵標準化器路徑
    feature_schema_path : str
        特徵架構檔案路徑
    mission : str
        任務名稱（TESS 或 Kepler）
    detrend_window : int
        去趨勢窗口大小
    verbose : bool
        是否顯示詳細資訊

    Returns:
    --------
    dict : 包含預測結果和中間資料的字典
    """
    result = {
        'tic_id': tic_id,
        'success': False,
        'error': None,
        'probability': None,
        'bls_period': None,
        'bls_snr': None,
        'bls_depth': None,
        'features': {},
        'lightcurve': None
    }

    try:
        # 1. 格式化 TIC ID
        if not tic_id.startswith("TIC") and not tic_id.startswith("KIC"):
            tic_id = f"TIC {tic_id}"

        if verbose:
            print(f"🔍 處理目標: {tic_id}")

        # 2. 下載光曲線
        if verbose:
            print("   下載光曲線...")

        if lk is None:
            raise ImportError("Lightkurve not available")

        search_result = lk.search_lightcurve(
            tic_id,
            mission=mission,
            author="SPOC" if mission == "TESS" else "Kepler"
        )

        if len(search_result) == 0:
            raise ValueError(f"找不到 {tic_id} 的光曲線資料")

        # 下載第一個結果
        lc = search_result[0].download()

        # 3. 清理和去趨勢
        if verbose:
            print("   清理和去趨勢...")

        lc_clean = lc.remove_nans()
        if len(lc_clean) < 100:
            raise ValueError("光曲線資料點太少（<100）")

        lc_flat = lc_clean.flatten(window_length=detrend_window)

        time = lc_flat.time.value
        flux = lc_flat.flux.value

        result['lightcurve'] = {
            'time': time.tolist(),
            'flux': flux.tolist(),
            'mission': mission,
            'sector': getattr(lc, 'sector', None),
            'n_points': len(time)
        }

        # 4. 執行 BLS 搜尋
        if verbose:
            print("   執行 BLS 搜尋...")

        bls_result = run_bls(time, flux)

        result['bls_period'] = bls_result.get('period')
        result['bls_snr'] = bls_result.get('snr')
        result['bls_depth'] = bls_result.get('depth')

        # 5. 提取特徵
        if verbose:
            print("   提取特徵...")

        features = extract_features(time, flux, bls_result, compute_advanced=True)
        result['features'] = features

        # 6. 載入模型和特徵架構
        if verbose:
            print("   載入模型...")

        model = joblib.load(model_path)

        # Handle scaler - it might be in the pipeline or separate
        if scaler_path and Path(scaler_path).exists():
            scaler = joblib.load(scaler_path)
        elif hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
            scaler = model.named_steps['scaler']
            if verbose:
                print("   使用 pipeline 中的標準化器")
        else:
            scaler = None
            if verbose:
                print("   ⚠️ 未找到標準化器，跳過特徵標準化")

        # Handle feature schema
        if feature_schema_path and Path(feature_schema_path).exists():
            with open(feature_schema_path, 'r') as f:
                schema = json.load(f)
            feature_order = schema['feature_order']
        else:
            # Use default feature order
            feature_order = [
                'bls_period', 'bls_duration', 'bls_depth', 'bls_snr', 'bls_power',
                'odd_even_mismatch', 'secondary_power_ratio', 'harmonic_delta_chisq',
                'periodicity_strength', 'transit_symmetry', 'odd_even_depth_diff',
                'phase_coverage', 'ingress_egress_asymmetry', 'v_shape_indicator'
            ]
            if verbose:
                print(f"   使用預設特徵順序 ({len(feature_order)} 個特徵)")

        # 7. 準備特徵向量
        feature_vector = []
        for feat_name in feature_order:
            value = features.get(feat_name, 0.0)
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            feature_vector.append(value)

        X = np.array(feature_vector).reshape(1, -1)

        # 8. 標準化和預測
        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        # 獲取預測機率
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_scaled)[0, 1]
        else:
            # 如果是校準模型
            prob = model.predict_proba(X)[0, 1]

        result['probability'] = float(prob)
        result['success'] = True

        if verbose:
            print(f"   ✅ 預測完成: {prob:.3f}")

    except Exception as e:
        result['error'] = str(e)
        if verbose:
            print(f"   ❌ 錯誤: {e}")

    return result


def predict_batch(
    tic_list: List[str],
    model_path: str = "model/ranker.joblib",
    scaler_path: str = "model/scaler.joblib",
    feature_schema_path: str = "model/feature_schema.json",
    mission: str = "TESS",
    n_jobs: int = 1,
    verbose: bool = True
) -> pd.DataFrame:
    """
    批次處理多個 TIC

    Parameters:
    -----------
    tic_list : list
        TIC ID 列表
    model_path : str
        模型路徑
    scaler_path : str
        標準化器路徑
    feature_schema_path : str
        特徵架構路徑
    mission : str
        任務名稱
    n_jobs : int
        並行處理數（目前僅支援 1）
    verbose : bool
        是否顯示進度

    Returns:
    --------
    pd.DataFrame : 包含所有結果的資料框
    """
    results = []

    for i, tic_id in enumerate(tic_list):
        if verbose:
            print(f"\n[{i+1}/{len(tic_list)}] 處理 {tic_id}")

        result = predict_from_tic(
            tic_id,
            model_path=model_path,
            scaler_path=scaler_path,
            feature_schema_path=feature_schema_path,
            mission=mission,
            verbose=verbose
        )

        # 簡化結果用於表格
        summary = {
            'tic_id': result['tic_id'],
            'probability': result['probability'],
            'bls_period': result['bls_period'],
            'bls_snr': result['bls_snr'],
            'bls_depth': result['bls_depth'],
            'success': result['success'],
            'error': result['error']
        }

        # 添加部分重要特徵
        for key in ['odd_even_depth_diff', 'transit_symmetry', 'periodicity_strength']:
            if key in result.get('features', {}):
                summary[key] = result['features'][key]

        results.append(summary)

    # 轉為 DataFrame
    df = pd.DataFrame(results)

    # 排序（按機率降序）
    if 'probability' in df.columns:
        df = df.sort_values('probability', ascending=False)

    if verbose:
        print(f"\n✅ 批次處理完成: {len(df)} 個目標")
        print(f"   成功: {df['success'].sum()}")
        print(f"   失敗: {(~df['success']).sum()}")

    return df


def check_gpu_availability():
    """
    檢查 GPU 可用性和類型

    Returns:
    --------
    dict : GPU 資訊
    """
    gpu_info = {
        'available': False,
        'device_name': None,
        'is_l4': False,
        'supports_bfloat16': False
    }

    try:
        import torch

        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['device_name'] = torch.cuda.get_device_name(0)

            # 檢查是否為 L4 GPU
            if 'L4' in gpu_info['device_name']:
                gpu_info['is_l4'] = True
                gpu_info['supports_bfloat16'] = True

            # 檢查 bfloat16 支援
            if hasattr(torch.cuda, 'is_bf16_supported'):
                gpu_info['supports_bfloat16'] = torch.cuda.is_bf16_supported()

    except ImportError:
        pass

    return gpu_info


def create_folded_lightcurve_plot(
    time: np.ndarray,
    flux: np.ndarray,
    period: float,
    t0: float = 0.0
) -> Dict[str, Any]:
    """
    創建摺疊光曲線資料

    Parameters:
    -----------
    time : np.ndarray
        時間序列
    flux : np.ndarray
        流量序列
    period : float
        週期
    t0 : float
        第一次凌日時刻

    Returns:
    --------
    dict : 摺疊光曲線資料
    """
    # 計算相位
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0

    # 分箱平均（用於繪圖）
    n_bins = 100
    phase_bins = np.linspace(-0.5, 0.5, n_bins + 1)
    binned_phase = []
    binned_flux = []
    binned_std = []

    for i in range(n_bins):
        mask = (phase >= phase_bins[i]) & (phase < phase_bins[i + 1])
        if np.sum(mask) > 0:
            binned_phase.append((phase_bins[i] + phase_bins[i + 1]) / 2)
            binned_flux.append(np.median(flux[mask]))
            binned_std.append(np.std(flux[mask]))

    return {
        'phase': phase.tolist(),
        'flux': flux.tolist(),
        'binned_phase': binned_phase,
        'binned_flux': binned_flux,
        'binned_std': binned_std,
        'period': period,
        't0': t0
    }


def save_inference_results(
    results_df: pd.DataFrame,
    output_path: str = "results/inference_results.csv",
    include_metadata: bool = True
) -> str:
    """
    儲存推論結果

    Parameters:
    -----------
    results_df : pd.DataFrame
        推論結果資料框
    output_path : str
        輸出路徑
    include_metadata : bool
        是否包含元資料

    Returns:
    --------
    str : 儲存路徑
    """
    # 建立輸出目錄
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 儲存 CSV
    results_df.to_csv(output_path, index=False)

    if include_metadata:
        # 儲存元資料
        import time
        metadata = {
            'n_targets': int(len(results_df)),
            'n_success': int(results_df['success'].sum()) if 'success' in results_df else 0,
            'inference_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'high_confidence': int(len(results_df[results_df['probability'] > 0.8])) if 'probability' in results_df else 0
        }

        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    return str(output_path)