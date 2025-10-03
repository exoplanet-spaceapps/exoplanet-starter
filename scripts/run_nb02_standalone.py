# Standalone execution of Notebook 02
# Auto-generated script to process all 11,979 samples

import sys
import os

# Add notebooks folder to path for module imports
notebooks_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks')
if notebooks_path not in sys.path:
    sys.path.insert(0, notebooks_path)

# Ensure UTF-8 output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ===== CELL 3 =====
# 環境設定與依賴安裝（Colab）
import sys, subprocess, pkgutil
import warnings
warnings.filterwarnings('ignore')

def pipi(*pkgs):
    """安裝套件的輔助函式"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

# 安裝必要套件（避免 numpy 2.0 相容性問題）
print("🚀 正在安裝依賴套件...")
try:
    import numpy as np
    import lightkurve as lk
    import transitleastsquares as tls
    print("✅ 基礎套件已安裝")
except Exception:
    pipi("numpy<2", "lightkurve", "astroquery", "scikit-learn", 
         "matplotlib", "wotan", "transitleastsquares")
    print("✅ 依賴套件安裝完成")

# 檢查 GPU 資訊
# 檢查 GPU 資訊（嘗試導入 torch）
try:
    import torch
except ImportError:
    torch = None

if torch is not None and torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🖥️ GPU 型號: {gpu_name}")
    print(f"   記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 如果是 NVIDIA L4，提供 BF16 優化建議
    if "L4" in gpu_name:
        print("💡 偵測到 NVIDIA L4 GPU - 支援高效能 BF16 運算")
        print("   建議在訓練時使用 torch.autocast('cuda', dtype=torch.bfloat16)")
else:
    try:
        # 使用 nvidia-smi 檢查 GPU
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            gpu_name = result.stdout.strip()
            print(f"🖥️ GPU 型號: {gpu_name}")
            if "L4" in gpu_name:
                print("💡 偵測到 NVIDIA L4 GPU - 支援高效能 BF16 運算")
    except:
        print("⚠️ 未偵測到 GPU，將使用 CPU 運算")

print("\n環境設定完成！")
# ===== CELL 4 =====
# 🔧 設定可重現性與日誌記錄 (2025 Best Practices)"""Phase 1: Critical Infrastructure- 設定隨機種子確保可重現性- 初始化日誌記錄系統- 記錄系統環境資訊"""import sysimport osfrom pathlib import Path# 確保 src 目錄在 Python 路徑中if IN_COLAB:    # Colab 環境：專案在 /content/exoplanet-starter    src_path = Path('/content/exoplanet-starter/src')else:    # 本地環境：向上一層找到專案根目錄    src_path = Path(__file__).parent.parent / 'src' if '__file__' in globals() else Path('../src').resolve()if src_path.exists() and str(src_path) not in sys.path:    sys.path.insert(0, str(src_path))    print(f"📂 已添加 src 路徑: {src_path}")# 導入工具模組try:    from utils import set_random_seeds, setup_logger, get_log_file_path, log_system_info    # 1️⃣ 設定隨機種子 (確保可重現性)    set_random_seeds(42)    # 2️⃣ 設定日誌記錄    log_file = get_log_file_path("02_bls_baseline", results_dir=Path("../results") if not IN_COLAB else Path("/content/exoplanet-starter/results"))    logger = setup_logger("02_bls_baseline", log_file=log_file, verbose=True)    # 3️⃣ 記錄系統資訊    logger.info("="*60)    logger.info("🚀 02_bls_baseline.ipynb 開始執行")    logger.info("="*60)    log_system_info(logger)    print("✅ 可重現性與日誌記錄設定完成")    print(f"   📝 日誌檔案: {log_file}")    print(f"   🎲 隨機種子: 42")except ImportError as e:    print(f"⚠️ 無法導入工具模組: {e}")    print("   跳過可重現性設定，繼續執行...")    # 如果導入失敗，創建一個簡單的 logger fallback    import logging    logger = logging.getLogger("02_bls_baseline")    logger.addHandler(logging.StreamHandler(sys.stdout))    logger.setLevel(logging.INFO)
# ===== CELL 6 =====
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
from transitleastsquares import transitleastsquares
from typing import Dict, Any, Tuple, Optional
import time

# 設定圖表風格
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

print("📚 套件導入完成")
print(f"   Lightkurve 版本: {lk.__version__}")
print(f"   NumPy 版本: {np.__version__}")
# ===== CELL 8 =====
# 🔧 載入已下載的資料集
"""
從 01_tap_download.ipynb 載入已處理的資料
使用 data_loader_colab.py 模組進行統一的資料載入
"""

# 導入資料載入模組
import data_loader_colab

# 執行完整的資料載入流程
# 自動處理 Colab/本地環境差異，從 GitHub 克隆資料（如需要）
sample_targets, datasets, data_dir, IN_COLAB = data_loader_colab.main()

# 資料載入完成，可以開始分析
print(f"\n✅ 資料載入完成！")
print(f"   📂 資料目錄: {data_dir}")
print(f"   🌍 環境: {'Google Colab' if IN_COLAB else '本地環境'}")
print(f"   📊 載入資料集: {len(datasets)} 個")
print(f"   🎯 分析樣本: {len(sample_targets)} 個目標")
print(f"\n準備開始 BLS/TLS 基線分析...")
# ===== CELL 9 =====
# 🎯 建立分析目標列表
"""
從載入的資料建立目標天體列表供 BLS/TLS 分析
"""

targets = []

# 從樣本中建立目標列表
for idx, row in sample_targets.iterrows():
    # 提取 TIC/KIC ID
    target_id = row.get('target_id', f'Unknown_{idx}')
    
    # 清理並格式化 ID
    if 'TIC' in str(target_id):
        clean_id = str(target_id).replace('TIC', '').strip()
        formatted_id = f"TIC {clean_id}"
        mission = "TESS"
    elif 'KIC' in str(target_id):
        clean_id = str(target_id).replace('KIC', '').strip() 
        formatted_id = f"KIC {clean_id}"
        mission = "Kepler"
    else:
        # 如果沒有明確標示，根據 ID 範圍判斷
        try:
            id_num = int(''.join(filter(str.isdigit, str(target_id))))
            if id_num > 100000000:  # 大於1億通常是TIC
                formatted_id = f"TIC {id_num}"
                mission = "TESS"
            else:  # 否則假設是KIC
                formatted_id = f"KIC {id_num}"
                mission = "Kepler"
        except:
            formatted_id = str(target_id)
            mission = "Unknown"
    
    # 建立目標字典
    target_dict = {
        "id": formatted_id,
        "mission": mission,
        "name": row.get('toi', row.get('target_name', target_id)),
        "description": f"{'正樣本 (行星候選)' if row['label'] == 1 else '負樣本 (False Positive)'}",
        "label": row['label'],
        "source": row.get('source', 'Unknown')
    }
    
    # 添加物理參數（如果有）
    if 'period' in row and pd.notna(row['period']):
        target_dict['known_period'] = float(row['period'])
    if 'depth' in row and pd.notna(row['depth']):
        target_dict['known_depth'] = float(row['depth'])
    
    targets.append(target_dict)

# 如果沒有從資料載入目標，使用預設目標
if len(targets) == 0:
    print("⚠️ 無法從資料集載入目標，使用預設目標")
    targets = [
        {"id": "TIC 25155310", "mission": "TESS", "name": "TOI-431", 
         "description": "擁有3顆已確認行星的K型矮星", "label": 1, "source": "default"},
        {"id": "TIC 307210830", "mission": "TESS", "name": "TOI-270",
         "description": "擁有3顆小型行星的M型矮星", "label": 1, "source": "default"},
        {"id": "KIC 11904151", "mission": "Kepler", "name": "Kepler-10",
         "description": "第一個被確認的岩石系外行星宿主恆星", "label": 1, "source": "default"}
    ]

print("🎯 分析目標：")
for i, target in enumerate(targets, 1):
    print(f"   {i}. {target['name']} ({target['id']}) - {target['mission']}")
    print(f"      {target['description']}")
    if 'known_period' in target:
        print(f"      已知週期: {target['known_period']:.3f} 天")
    if 'known_depth' in target:
        print(f"      已知深度: {target['known_depth']:.0f} ppm")
    print()

print(f"✅ 建立完成，共 {len(targets)} 個分析目標")
# ===== CELL 11 =====
def download_and_process_lightcurve(
    target_id: str, 
    mission: str, 
    author: str = "SPOC",
    cadence: str = "short"
) -> Tuple[lk.LightCurve, lk.LightCurve, Dict[str, Any]]:
    """
    下載並處理光曲線資料
    
    Parameters:
    -----------
    target_id : str
        目標天體識別碼（TIC/KIC）
    mission : str
        任務名稱（TESS/Kepler）
    author : str
        資料提供者（SPOC/PDCSAP）
    cadence : str
        觀測頻率（short/long）
    
    Returns:
    --------
    tuple : (原始光曲線, 去趨勢光曲線, metadata字典)
    """
    print(f"\n📡 正在下載 {target_id} 的光曲線...")
    
    # 搜尋並下載光曲線
    search_result = lk.search_lightcurve(
        target_id, 
        mission=mission, 
        author=author if mission == "TESS" else None,
        cadence=cadence
    )
    
    if len(search_result) == 0:
        raise ValueError(f"未找到 {target_id} 的光曲線資料")
    
    print(f"   找到 {len(search_result)} 個光曲線檔案")
    
    # 下載第一個sector/quarter的資料
    lc_collection = search_result[0].download()
    
    # 如果是collection，取第一個光曲線
    if hasattr(lc_collection, '__iter__'):
        lc_raw = lc_collection[0]
    else:
        lc_raw = lc_collection
        
    # 記錄metadata
    metadata = {
        "target_id": target_id,
        "mission": mission,
        "sector" if mission == "TESS" else "quarter": lc_raw.meta.get('SECTOR', lc_raw.meta.get('QUARTER', 'N/A')),
        "exposure_time": lc_raw.meta.get('EXPOSURE', 'N/A'),
        "n_points_raw": len(lc_raw.time),
    }
    
    print(f"   ✅ 下載完成：{metadata['n_points_raw']} 個資料點")
    
    # 清理資料：移除NaN值
    lc_clean = lc_raw.remove_nans()
    
    # 去趨勢處理
    print(f"   🔧 正在進行去趨勢處理...")
    lc_flat = lc_clean.flatten(window_length=401)
    
    metadata['n_points_clean'] = len(lc_clean.time)
    metadata['n_points_flat'] = len(lc_flat.time)
    metadata['removed_points'] = metadata['n_points_raw'] - metadata['n_points_clean']
    
    print(f"   ✅ 去趨勢完成：保留 {metadata['n_points_flat']} 個資料點")
    
    return lc_clean, lc_flat, metadata
# ===== CELL 13 =====
# 儲存處理結果
processed_data = {}

for target in targets:
    try:
        lc_clean, lc_flat, metadata = download_and_process_lightcurve(
            target["id"],
            target["mission"],
            author="SPOC" if target["mission"] == "TESS" else None
        )
        
        processed_data[target["id"]] = {
            "target": target,
            "lc_clean": lc_clean,
            "lc_flat": lc_flat,
            "metadata": metadata
        }
        
    except Exception as e:
        print(f"   ❌ 處理 {target['id']} 時發生錯誤: {str(e)}")
        continue

print(f"\n✅ 成功處理 {len(processed_data)} 個目標")
# ===== CELL 15 =====
def plot_raw_vs_detrended(data_dict: Dict[str, Any]):
    """
    繪製原始與去趨勢光曲線對比圖
    """
    target = data_dict["target"]
    lc_clean = data_dict["lc_clean"]
    lc_flat = data_dict["lc_flat"]
    metadata = data_dict["metadata"]
    
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(3, 1, height_ratios=[1, 1, 0.8], hspace=0.3)
    
    # 原始光曲線
    ax1 = fig.add_subplot(gs[0])
    lc_clean.plot(ax=ax1, color='blue', alpha=0.7, label='原始光曲線')
    ax1.set_title(f"{target['name']} ({target['id']}) - 原始光曲線", fontsize=12, fontweight='bold')
    ax1.set_ylabel('相對流量 (e⁻/s)', fontsize=10)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 去趨勢光曲線
    ax2 = fig.add_subplot(gs[1])
    lc_flat.plot(ax=ax2, color='green', alpha=0.7, label='去趨勢光曲線')
    ax2.set_title('去趨勢後光曲線（window_length=401）', fontsize=12, fontweight='bold')
    ax2.set_ylabel('標準化流量', fontsize=10)
    ax2.set_xlabel('時間 (BTJD)', fontsize=10)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 直方圖比較
    ax3 = fig.add_subplot(gs[2])
    
    # 計算標準化的流量值
    flux_clean_norm = (lc_clean.flux - np.nanmean(lc_clean.flux)) / np.nanstd(lc_clean.flux)
    flux_flat_norm = (lc_flat.flux - np.nanmean(lc_flat.flux)) / np.nanstd(lc_flat.flux)
    
    ax3.hist(flux_clean_norm, bins=50, alpha=0.5, color='blue', label='原始', density=True)
    ax3.hist(flux_flat_norm, bins=50, alpha=0.5, color='green', label='去趨勢', density=True)
    ax3.set_xlabel('標準化流量', fontsize=10)
    ax3.set_ylabel('機率密度', fontsize=10)
    ax3.set_title('流量分佈比較', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 添加文字說明
    textstr = f"""資料統計:
原始資料點: {metadata['n_points_raw']:,}
清理後: {metadata['n_points_clean']:,}
移除NaN: {metadata['removed_points']:,}
{'Sector' if metadata['mission'] == 'TESS' else 'Quarter'}: {metadata.get('sector', metadata.get('quarter', 'N/A'))}
"""
    ax3.text(0.02, 0.98, textstr, transform=ax3.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f"{target['description']}", fontsize=11, y=1.02)
    plt.tight_layout()
    plt.show()
    
    return fig
# ===== CELL 16 =====
# 繪製所有目標的對比圖
for target_id, data in processed_data.items():
    print(f"\n📊 繪製 {data['target']['name']} 的光曲線對比圖...")
    fig = plot_raw_vs_detrended(data)
    
    # 說明文字
    print(f"""
    💡 說明：
    - 原始光曲線顯示了儀器效應造成的長期趨勢
    - 去趨勢處理保留了短週期變化（如行星凌日）
    - 流量分佈圖顯示去趨勢後的資料更接近常態分佈
    """)
# ===== CELL 18 =====
def run_bls_search(
    lc: lk.LightCurve,
    min_period: float = 0.5,
    max_period: float = 20.0,
    frequency_factor: float = 5.0
) -> Dict[str, Any]:
    """
    執行 BLS 週期搜尋
    
    Parameters:
    -----------
    lc : lightkurve.LightCurve
        輸入光曲線
    min_period : float
        最小搜尋週期（天）
    max_period : float
        最大搜尋週期（天）
    frequency_factor : float
        頻率解析度因子
    
    Returns:
    --------
    dict : BLS 結果字典
    """
    print(f"   🔍 執行 BLS 搜尋 ({min_period:.1f} - {max_period:.1f} 天)...")
    
    start_time = time.time()
    
    # 執行 BLS
    bls = lc.to_periodogram(
        method="bls",
        minimum_period=min_period,
        maximum_period=max_period,
        frequency_factor=frequency_factor
    )
    
    # 提取最強峰值的參數
    period = bls.period_at_max_power
    t0 = bls.transit_time_at_max_power
    duration = bls.duration_at_max_power
    depth = bls.depth_at_max_power
    snr = bls.max_power
    
    elapsed_time = time.time() - start_time
    
    results = {
        "periodogram": bls,
        "period": period.value if hasattr(period, 'value') else period,
        "t0": t0.value if hasattr(t0, 'value') else t0,
        "duration": duration.value if hasattr(duration, 'value') else duration,
        "depth": depth.value if hasattr(depth, 'value') else depth,
        "snr": snr.value if hasattr(snr, 'value') else snr,
        "elapsed_time": elapsed_time
    }
    
    print(f"   ✅ BLS 完成（耗時 {elapsed_time:.2f} 秒）")
    print(f"      最佳週期: {results['period']:.4f} 天")
    print(f"      SNR: {results['snr']:.2f}")
    print(f"      深度: {results['depth']*1e6:.0f} ppm")
    
    return results
# ===== CELL 20 =====
def run_tls_search(
    lc: lk.LightCurve,
    min_period: float = 0.5,
    max_period: float = 20.0
) -> Dict[str, Any]:
    """
    執行 TLS 週期搜尋
    
    Parameters:
    -----------
    lc : lightkurve.LightCurve
        輸入光曲線
    min_period : float
        最小搜尋週期（天）
    max_period : float
        最大搜尋週期（天）
    
    Returns:
    --------
    dict : TLS 結果字典
    """
    print(f"   🔍 執行 TLS 搜尋 ({min_period:.1f} - {max_period:.1f} 天)...")
    
    start_time = time.time()
    
    # 準備 TLS 輸入
    time_array = lc.time.value if hasattr(lc.time, 'value') else np.array(lc.time)
    flux_array = lc.flux.value if hasattr(lc.flux, 'value') else np.array(lc.flux)
    
    # 初始化 TLS
    model = transitleastsquares(time_array, flux_array)
    
    # 執行搜尋
    tls_results = model.power(
        period_min=min_period,
        period_max=max_period,
        show_progress_bar=False,
        use_threads=4
    )
    
    elapsed_time = time.time() - start_time
    
    results = {
        "tls_object": tls_results,
        "period": tls_results.period,
        "t0": tls_results.T0,
        "duration": tls_results.duration,
        "depth": tls_results.depth,
        "snr": tls_results.SDE,  # Signal Detection Efficiency
        "elapsed_time": elapsed_time,
        "periods": tls_results.periods,
        "power": tls_results.power
    }
    
    print(f"   ✅ TLS 完成（耗時 {elapsed_time:.2f} 秒）")
    print(f"      最佳週期: {results['period']:.4f} 天")
    print(f"      SDE: {results['snr']:.2f}")
    print(f"      深度: {results['depth']*1e6:.0f} ppm")
    
    return results
# ===== CELL 22 =====
# 儲存所有搜尋結果
search_results = {}

for target_id, data in processed_data.items():
    print(f"\n🚀 分析 {data['target']['name']} ({target_id})...")
    
    # 執行 BLS
    bls_results = run_bls_search(
        data['lc_flat'],
        min_period=0.5,
        max_period=20.0
    )
    
    # 執行 TLS
    tls_results = run_tls_search(
        data['lc_flat'],
        min_period=0.5,
        max_period=20.0
    )
    
    search_results[target_id] = {
        "bls": bls_results,
        "tls": tls_results,
        "target": data['target'],
        "lc_flat": data['lc_flat']
    }
    
print("\n✅ 所有目標的 BLS/TLS 搜尋完成！")
# ===== CELL 24 =====
def plot_bls_tls_comparison(search_result: Dict[str, Any]):
    """
    繪製 BLS 與 TLS 結果對比圖
    """
    target = search_result['target']
    bls_result = search_result['bls']
    tls_result = search_result['tls']
    lc_flat = search_result['lc_flat']
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.25)
    
    # BLS 功率譜
    ax1 = fig.add_subplot(gs[0, 0])
    bls_result['periodogram'].plot(ax=ax1, color='blue')
    ax1.set_title('BLS 功率譜', fontsize=12, fontweight='bold')
    ax1.axvline(bls_result['period'], color='red', linestyle='--', alpha=0.7, 
               label=f"P = {bls_result['period']:.3f} d")
    ax1.legend()
    ax1.set_ylabel('BLS Power')
    ax1.grid(True, alpha=0.3)
    
    # TLS 功率譜
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(tls_result['periods'], tls_result['power'], 'g-', lw=1)
    ax2.set_title('TLS 功率譜', fontsize=12, fontweight='bold')
    ax2.axvline(tls_result['period'], color='red', linestyle='--', alpha=0.7,
               label=f"P = {tls_result['period']:.3f} d")
    ax2.legend()
    ax2.set_xlabel('週期 (天)')
    ax2.set_ylabel('SDE (Signal Detection Efficiency)')
    ax2.set_xlim(0.5, 20)
    ax2.grid(True, alpha=0.3)
    
    # BLS 摺疊光曲線
    ax3 = fig.add_subplot(gs[1, 0])
    folded_bls = lc_flat.fold(period=bls_result['period'], epoch_time=bls_result['t0'])
    folded_bls.scatter(ax=ax3, s=1, color='blue', alpha=0.3)
    folded_bls.bin(time_bin_size=0.001).plot(
        ax=ax3, color='darkblue', markersize=4, label='Binned'
    )
    ax3.set_title(f"BLS 摺疊光曲線 (P={bls_result['period']:.3f} d)", fontsize=12)
    ax3.set_xlabel('相位')
    ax3.set_ylabel('標準化流量')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # TLS 摺疊光曲線
    ax4 = fig.add_subplot(gs[1, 1])
    folded_tls = lc_flat.fold(period=tls_result['period'], epoch_time=tls_result['t0'])
    folded_tls.scatter(ax=ax4, s=1, color='green', alpha=0.3)
    folded_tls.bin(time_bin_size=0.001).plot(
        ax=ax4, color='darkgreen', markersize=4, label='Binned'
    )
    ax4.set_title(f"TLS 摺疊光曲線 (P={tls_result['period']:.3f} d)", fontsize=12)
    ax4.set_xlabel('相位')
    ax4.set_ylabel('標準化流量')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 參數比較表
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')
    
    # 建立比較表格
    comparison_data = [
        ['參數', 'BLS', 'TLS', '差異 (%)'],
        ['週期 (天)', f"{bls_result['period']:.4f}", f"{tls_result['period']:.4f}", 
         f"{100*(tls_result['period']-bls_result['period'])/bls_result['period']:.1f}%"],
        ['SNR/SDE', f"{bls_result['snr']:.2f}", f"{tls_result['snr']:.2f}",
         f"{100*(tls_result['snr']-bls_result['snr'])/bls_result['snr']:.1f}%"],
        ['深度 (ppm)', f"{bls_result['depth']*1e6:.0f}", f"{tls_result['depth']*1e6:.0f}",
         f"{100*(tls_result['depth']-bls_result['depth'])/bls_result['depth']:.1f}%"],
        ['持續時間 (小時)', f"{bls_result['duration']*24:.2f}", f"{tls_result['duration']*24:.2f}",
         f"{100*(tls_result['duration']-bls_result['duration'])/bls_result['duration']:.1f}%"],
        ['運算時間 (秒)', f"{bls_result['elapsed_time']:.2f}", f"{tls_result['elapsed_time']:.2f}",
         f"{100*(tls_result['elapsed_time']-bls_result['elapsed_time'])/bls_result['elapsed_time']:.1f}%"]
    ]
    
    table = ax5.table(cellText=comparison_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 設定表格樣式
    for i in range(len(comparison_data)):
        for j in range(len(comparison_data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f1f1f2')
            cell.set_edgecolor('white')
    
    plt.suptitle(f"{target['name']} ({target['id']}) - BLS vs TLS 比較", 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    return fig
# ===== CELL 25 =====
# 繪製所有目標的 BLS vs TLS 比較圖
for target_id, result in search_results.items():
    print(f"\n📊 繪製 {result['target']['name']} 的 BLS vs TLS 比較圖...")
    fig = plot_bls_tls_comparison(result)
# ===== CELL 27 =====
# 生成總結報告
print("="*80)
print("📋 BLS vs TLS 總結報告")
print("="*80)

summary_data = []

for target_id, result in search_results.items():
    target = result['target']
    bls = result['bls']
    tls = result['tls']
    
    print(f"\n🎯 {target['name']} ({target_id})")
    print(f"   {target['description']}")
    print("\n   方法比較：")
    print(f"   {'方法':<10} {'週期(天)':<12} {'SNR/SDE':<10} {'深度(ppm)':<12} {'時間(秒)':<10}")
    print("   " + "-"*60)
    print(f"   {'BLS':<10} {bls['period']:<12.4f} {bls['snr']:<10.2f} "
          f"{bls['depth']*1e6:<12.1f} {bls['elapsed_time']:<10.2f}")
    print(f"   {'TLS':<10} {tls['period']:<12.4f} {tls['snr']:<10.2f} "
          f"{tls['depth']*1e6:<12.1f} {tls['elapsed_time']:<10.2f}")
    
    # 計算差異
    period_diff = abs(tls['period'] - bls['period']) / bls['period'] * 100
    snr_diff = (tls['snr'] - bls['snr']) / bls['snr'] * 100
    
    print(f"\n   關鍵差異：")
    print(f"   • 週期差異: {period_diff:.2f}%")
    print(f"   • SNR 改善: {snr_diff:+.1f}%")
    print(f"   • TLS 運算時間: {tls['elapsed_time']/bls['elapsed_time']:.1f}x BLS")
    
    summary_data.append({
        'target': target['name'],
        'period_diff_%': period_diff,
        'snr_improvement_%': snr_diff,
        'time_ratio': tls['elapsed_time']/bls['elapsed_time']
    })
# ===== CELL 28 =====
# 總體統計
print("\n" + "="*80)
print("📊 總體統計分析")
print("="*80)

if summary_data:
    avg_period_diff = np.mean([d['period_diff_%'] for d in summary_data])
    avg_snr_improvement = np.mean([d['snr_improvement_%'] for d in summary_data])
    avg_time_ratio = np.mean([d['time_ratio'] for d in summary_data])
    
    print(f"""
📌 主要發現：

1. **週期估計精度**：
   - BLS 與 TLS 的週期估計平均差異: {avg_period_diff:.2f}%
   - 兩種方法對週期的估計高度一致

2. **偵測靈敏度**：
   - TLS 相對 BLS 的平均 SNR 改善: {avg_snr_improvement:+.1f}%
   - TLS 使用更真實的凌日模型，通常能獲得更高的偵測靈敏度

3. **運算效率**：
   - TLS 平均運算時間是 BLS 的 {avg_time_ratio:.1f} 倍
   - BLS 更快速，適合初步篩選
   - TLS 更精確，適合確認候選體

4. **方法選擇建議**：
   - **BLS**：快速搜尋、大量資料初步篩選、即時分析
   - **TLS**：精確測量、候選體確認、小型行星偵測
   - **組合策略**：先用 BLS 快速篩選，再用 TLS 精確分析

5. **技術差異**：
   - **BLS**：假設箱型（方形）凌日模型，計算簡單快速
   - **TLS**：使用真實凌日模型（含邊緣變暗），考慮恆星物理
    """)
# ===== CELL 31 =====
def extract_bls_tls_features(search_results):
    """
    從 BLS/TLS 搜尋結果提取機器學習特徵
    
    Parameters:
    -----------
    search_results : dict
        包含 BLS 和 TLS 結果的字典
    
    Returns:
    --------
    dict : 特徵字典
    """
    features = {}
    
    # 提取目標資訊
    if 'target' in search_results:
        target = search_results['target']
        features['target_id'] = target.get('id', '')
        features['target_name'] = target.get('name', '')
        features['label'] = target.get('label', -1)
        features['source'] = target.get('source', '')
        features['known_period'] = target.get('known_period', np.nan)
        features['known_depth'] = target.get('known_depth', np.nan)
    
    # BLS 特徵
    if 'bls' in search_results:
        bls = search_results['bls']
        features['bls_period'] = bls['period']
        features['bls_t0'] = bls['t0']
        features['bls_duration_hours'] = bls['duration'] * 24
        features['bls_depth_ppm'] = bls['depth'] * 1e6
        features['bls_snr'] = bls['snr']
        
        # 計算額外的 BLS 特徵
        if bls['period'] > 0:
            features['bls_duration_phase'] = bls['duration'] / bls['period']  # 相位持續時間
    
    # TLS 特徵
    if 'tls' in search_results:
        tls = search_results['tls']
        features['tls_period'] = tls['period']
        features['tls_t0'] = tls['t0']
        features['tls_duration_hours'] = tls['duration'] * 24
        features['tls_depth_ppm'] = tls['depth'] * 1e6
        features['tls_sde'] = tls['snr']  # Signal Detection Efficiency
        
        # 計算額外的 TLS 特徵
        if tls['period'] > 0:
            features['tls_duration_phase'] = tls['duration'] / tls['period']
    
    # 計算 BLS vs TLS 比較特徵
    if 'bls' in search_results and 'tls' in search_results:
        bls = search_results['bls']
        tls = search_results['tls']
        
        # 週期一致性
        if bls['period'] > 0:
            features['period_ratio'] = tls['period'] / bls['period']
            features['period_diff_pct'] = abs(tls['period'] - bls['period']) / bls['period'] * 100
        
        # 深度一致性
        if bls['depth'] > 0:
            features['depth_ratio'] = tls['depth'] / bls['depth']
            features['depth_diff_pct'] = abs(tls['depth'] - bls['depth']) / bls['depth'] * 100
        
        # SNR 比較
        if bls['snr'] > 0:
            features['snr_ratio'] = tls['snr'] / bls['snr']
            features['snr_improvement'] = (tls['snr'] - bls['snr']) / bls['snr'] * 100
    
    # 添加資料品質標記
    features['has_bls'] = 1 if 'bls' in search_results else 0
    features['has_tls'] = 1 if 'tls' in search_results else 0
    
    return features

# 提取所有目標的特徵
all_features = []

for target_id, result in search_results.items():
    features = extract_bls_tls_features(result)
    all_features.append(features)

# 轉換為 DataFrame
features_df = pd.DataFrame(all_features)

print("📊 提取的特徵統計：")
print(f"   樣本數: {len(features_df)}")
print(f"   特徵數: {len(features_df.columns)}")
print(f"   正樣本: {(features_df['label'] == 1).sum()}")
print(f"   負樣本: {(features_df['label'] == 0).sum()}")

# 顯示特徵列表
print("\n📝 特徵列表：")
feature_cols = [col for col in features_df.columns if col not in ['target_id', 'target_name', 'source']]
for i, col in enumerate(feature_cols, 1):
    if not features_df[col].isna().all():
        print(f"   {i:2}. {col}: {features_df[col].dtype}, "
              f"非空值: {features_df[col].notna().sum()}/{len(features_df)}")

# 顯示前幾筆資料
print("\n🔍 特徵樣本（前3筆）：")
display_cols = ['target_name', 'label', 'bls_period', 'bls_snr', 'tls_period', 'tls_sde']
available_cols = [col for col in display_cols if col in features_df.columns]
print(features_df[available_cols].head(3).to_string(index=False))
# ===== CELL 32 =====
# 儲存特徵到檔案
output_dir = Path("../data")
output_dir.mkdir(parents=True, exist_ok=True)

# 儲存特徵 CSV
features_file = output_dir / "bls_tls_features.csv"
features_df.to_csv(features_file, index=False)
print(f"\n💾 特徵已儲存至: {features_file}")

# 儲存特徵統計
stats = {
    'n_samples': len(features_df),
    'n_features': len(features_df.columns),
    'n_positive': int((features_df['label'] == 1).sum()),
    'n_negative': int((features_df['label'] == 0).sum()),
    'features': list(features_df.columns),
    'bls_features': [col for col in features_df.columns if col.startswith('bls_')],
    'tls_features': [col for col in features_df.columns if col.startswith('tls_')],
    'comparison_features': ['period_ratio', 'depth_ratio', 'snr_ratio', 'period_diff_pct', 'depth_diff_pct', 'snr_improvement']
}

# 儲存統計資訊
import json
stats_file = output_dir / "bls_tls_features_stats.json"
with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"📊 統計資訊已儲存至: {stats_file}")

# 建立特徵重要性初步分析（如果有足夠樣本）
if len(features_df) >= 10 and features_df['label'].nunique() == 2:
    print("\n🔬 特徵重要性初步分析：")
    
    # 計算各特徵與標籤的相關性
    numerical_features = features_df.select_dtypes(include=[np.number]).columns
    correlations = {}
    
    for col in numerical_features:
        if col != 'label' and features_df[col].notna().sum() > 5:
            corr = features_df[[col, 'label']].corr()['label'][col]
            if not pd.isna(corr):
                correlations[col] = corr
    
    # 排序並顯示前10個最相關的特徵
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
    
    print("\n   與標籤最相關的特徵（相關係數）：")
    for feat, corr in sorted_corr:
        print(f"   • {feat}: {corr:+.3f}")
    
    # 比較正負樣本的特徵差異
    print("\n   正負樣本特徵差異：")
    for col in ['bls_snr', 'tls_sde', 'bls_depth_ppm', 'tls_depth_ppm']:
        if col in features_df.columns:
            pos_mean = features_df[features_df['label'] == 1][col].mean()
            neg_mean = features_df[features_df['label'] == 0][col].mean()
            if not pd.isna(pos_mean) and not pd.isna(neg_mean):
                diff_pct = (pos_mean - neg_mean) / abs(neg_mean) * 100 if neg_mean != 0 else 0
                print(f"   • {col}:")
                print(f"     正樣本平均: {pos_mean:.2f}")
                print(f"     負樣本平均: {neg_mean:.2f}")
                print(f"     差異: {diff_pct:+.1f}%")

print("\n✅ BLS/TLS 特徵提取完成！")
print("   可使用這些特徵進行機器學習訓練（03_injection_train.ipynb）")
# ===== CELL 33 =====
# 建立結果摘要 DataFrame
import pandas as pd

results_list = []
for target_id, result in search_results.items():
    target = result['target']
    bls = result['bls']
    tls = result['tls']
    
    results_list.append({
        'Target': target['name'],
        'ID': target_id,
        'Mission': target['mission'],
        'BLS_Period_days': bls['period'],
        'BLS_SNR': bls['snr'],
        'BLS_Depth_ppm': bls['depth']*1e6,
        'BLS_Duration_hours': bls['duration']*24,
        'TLS_Period_days': tls['period'],
        'TLS_SDE': tls['snr'],
        'TLS_Depth_ppm': tls['depth']*1e6,
        'TLS_Duration_hours': tls['duration']*24,
        'Period_Difference_%': abs(tls['period']-bls['period'])/bls['period']*100,
        'SNR_Improvement_%': (tls['snr']-bls['snr'])/bls['snr']*100
    })

results_df = pd.DataFrame(results_list)

print("\n📊 結果摘要表：")
print("\n", results_df.to_string(index=False))

# 可選：儲存到 CSV
# results_df.to_csv('bls_tls_results.csv', index=False)
# print("\n💾 結果已儲存至 bls_tls_results.csv")
# ===== CELL 35 =====
# 🚀 執行 GitHub Push
# 取消註解下面這行來執行推送:
# ultimate_push_to_github_02()

print("📋 BLS/TLS 基線分析完成！")
print("💡 請在需要推送結果時執行上面的 ultimate_push_to_github_02() 函數")
# ===== CELL 39 =====
# 儲存增強特徵到檔案
output_dir = Path("../data")
output_dir.mkdir(parents=True, exist_ok=True)

# 儲存增強特徵 CSV
enhanced_features_file = output_dir / "bls_tls_features_enhanced.csv"
enhanced_features_df.to_csv(enhanced_features_file, index=False)
print(f"\n💾 增強特徵已儲存至: {enhanced_features_file}")

# 儲存特徵統計與說明
enhanced_stats = {
    'n_samples': len(enhanced_features_df),
    'n_features': len(enhanced_features_df.columns),
    'n_positive': int((enhanced_features_df['label'] == 1).sum()),
    'n_negative': int((enhanced_features_df['label'] == 0).sum()),
    'feature_categories': {
        'basic_info': ['target_id', 'target_name', 'label', 'source', 'known_period', 'known_depth'],
        'bls_features': [col for col in enhanced_features_df.columns if col.startswith('bls_')],
        'tls_features': [col for col in enhanced_features_df.columns if col.startswith('tls_')],
        'comparison_features': ['period_ratio', 'depth_ratio', 'snr_ratio', 'period_diff_pct', 'depth_diff_pct', 'snr_improvement'],
        'detrending_features': [col for col in enhanced_features_df.columns if 'detrend' in col or col.endswith('_snr')],
        'odd_even_features': ['odd_depth_ppm', 'even_depth_ppm', 'odd_even_ratio', 'odd_even_diff_ppm'],
        'shape_features': ['transit_curvature', 'transit_symmetry', 'transit_points']
    },
    'phase_5_features': [col for col in enhanced_features_df.columns if 'detrend' in col or (col.endswith('_snr') and 'wotan' in col)],
    'phase_6_features': ['odd_depth_ppm', 'even_depth_ppm', 'odd_even_ratio', 'odd_even_diff_ppm', 
                         'transit_curvature', 'transit_symmetry', 'transit_points']
}

# 儲存統計資訊
import json
enhanced_stats_file = output_dir / "bls_tls_features_enhanced_stats.json"
with open(enhanced_stats_file, 'w') as f:
    json.dump(enhanced_stats, f, indent=2)
print(f"📊 增強特徵統計已儲存至: {enhanced_stats_file}")

# 顯示各類別特徵數量
print("\n📋 特徵分類統計：")
for category, features_list in enhanced_stats['feature_categories'].items():
    print(f"   • {category}: {len(features_list)} 個特徵")

print(f"\n🌟 Phase 5 新增特徵: {len(enhanced_stats['phase_5_features'])} 個")
print(f"   {enhanced_stats['phase_5_features']}")

print(f"\n🎯 Phase 6 新增特徵: {len(enhanced_stats['phase_6_features'])} 個")
print(f"   {enhanced_stats['phase_6_features']}")

print("\n✅ Phase 5 & 6 完成！")
print("   所有增強特徵已準備完成，可用於 Phase 3 監督學習訓練")
# ===== CELL 40 =====
# 🎯 Phase 6: Advanced BLS Metrics Extraction
"""
提取額外的 BLS 指標與特徵
"""

print("="*60)
print("🎯 Phase 6: Advanced BLS Metrics Extraction")
print("="*60)

def calculate_odd_even_depth(lc: lk.LightCurve, period: float, t0: float, duration: float) -> Dict[str, float]:
    """
    計算奇偶次凌日深度差異（用於檢測假陽性，如雙星系統）
    
    Parameters:
    -----------
    lc : lightkurve.LightCurve
        去趨勢光曲線
    period : float
        凌日週期
    t0 : float
        第一次凌日時間
    duration : float
        凌日持續時間
    
    Returns:
    --------
    dict : 包含奇偶深度與比率的字典
    """
    try:
        time_array = lc.time.value if hasattr(lc.time, 'value') else np.array(lc.time)
        flux_array = lc.flux.value if hasattr(lc.flux, 'value') else np.array(lc.flux)
        
        # 計算每個資料點所屬的週期編號
        phase = (time_array - t0) / period
        cycle_number = np.floor(phase)
        
        # 分離奇數和偶數週期
        odd_mask = (cycle_number % 2 == 1) & (np.abs(phase - cycle_number) < duration / period)
        even_mask = (cycle_number % 2 == 0) & (np.abs(phase - cycle_number) < duration / period)
        
        # 計算深度（相對於 1.0）
        if np.sum(odd_mask) > 0 and np.sum(even_mask) > 0:
            odd_depth = 1.0 - np.median(flux_array[odd_mask])
            even_depth = 1.0 - np.median(flux_array[even_mask])
            
            # 計算差異比率
            if even_depth > 0:
                depth_ratio = odd_depth / even_depth
            else:
                depth_ratio = np.nan
            
            return {
                'odd_depth_ppm': odd_depth * 1e6,
                'even_depth_ppm': even_depth * 1e6,
                'odd_even_ratio': depth_ratio,
                'odd_even_diff_ppm': (odd_depth - even_depth) * 1e6
            }
        else:
            return {
                'odd_depth_ppm': np.nan,
                'even_depth_ppm': np.nan,
                'odd_even_ratio': np.nan,
                'odd_even_diff_ppm': np.nan
            }
    except Exception as e:
        print(f"      ⚠️ 計算奇偶深度失敗: {e}")
        return {
            'odd_depth_ppm': np.nan,
            'even_depth_ppm': np.nan,
            'odd_even_ratio': np.nan,
            'odd_even_diff_ppm': np.nan
        }

def calculate_transit_shape_metrics(lc: lk.LightCurve, period: float, t0: float, duration: float) -> Dict[str, float]:
    """
    計算凌日形狀指標
    
    Parameters:
    -----------
    lc : lightkurve.LightCurve
        去趨勢光曲線
    period : float
        凌日週期
    t0 : float
        第一次凌日時間
    duration : float
        凌日持續時間
    
    Returns:
    --------
    dict : 包含形狀指標的字典
    """
    try:
        # 摺疊光曲線
        folded_lc = lc.fold(period=period, epoch_time=t0)
        
        time_array = folded_lc.time.value if hasattr(folded_lc.time, 'value') else np.array(folded_lc.time)
        flux_array = folded_lc.flux.value if hasattr(folded_lc.flux, 'value') else np.array(folded_lc.flux)
        
        # 選擇凌日區域
        transit_mask = np.abs(time_array) < duration / 2
        
        if np.sum(transit_mask) > 10:  # 至少需要10個點
            transit_flux = flux_array[transit_mask]
            transit_time = time_array[transit_mask]
            
            # 計算 V-shape vs U-shape (曲率)
            # 簡化版：計算最深點附近的曲率
            min_idx = np.argmin(transit_flux)
            if min_idx > 0 and min_idx < len(transit_flux) - 1:
                curvature = (transit_flux[min_idx-1] + transit_flux[min_idx+1] - 2*transit_flux[min_idx])
            else:
                curvature = np.nan
            
            # 計算對稱性（左右半部的差異）
            mid_idx = len(transit_flux) // 2
            left_mean = np.mean(transit_flux[:mid_idx])
            right_mean = np.mean(transit_flux[mid_idx:])
            symmetry = abs(left_mean - right_mean) / np.std(transit_flux)
            
            return {
                'transit_curvature': curvature,
                'transit_symmetry': symmetry,
                'transit_points': int(np.sum(transit_mask))
            }
        else:
            return {
                'transit_curvature': np.nan,
                'transit_symmetry': np.nan,
                'transit_points': int(np.sum(transit_mask))
            }
    except Exception as e:
        print(f"      ⚠️ 計算凌日形狀失敗: {e}")
        return {
            'transit_curvature': np.nan,
            'transit_symmetry': np.nan,
            'transit_points': 0
        }

def extract_enhanced_bls_features(
    search_result: Dict[str, Any],
    detrending_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    提取增強的 BLS 特徵（包含 Phase 5 和 Phase 6）
    
    Parameters:
    -----------
    search_result : dict
        BLS/TLS 搜尋結果
    detrending_result : dict
        去趨勢方法比較結果
    
    Returns:
    --------
    dict : 增強特徵字典
    """
    features = {}
    
    # 基本資訊
    target = search_result['target']
    features['target_id'] = target.get('id', '')
    features['target_name'] = target.get('name', '')
    features['label'] = target.get('label', -1)
    features['source'] = target.get('source', '')
    features['known_period'] = target.get('known_period', np.nan)
    features['known_depth'] = target.get('known_depth', np.nan)
    
    # BLS 基本特徵
    if 'bls' in search_result:
        bls = search_result['bls']
        features['bls_period'] = bls['period']
        features['bls_t0'] = bls['t0']
        features['bls_duration_hours'] = bls['duration'] * 24
        features['bls_depth_ppm'] = bls['depth'] * 1e6
        features['bls_snr'] = bls['snr']
        features['bls_duration_phase'] = bls['duration'] / bls['period'] if bls['period'] > 0 else np.nan
    
    # TLS 基本特徵
    if 'tls' in search_result:
        tls = search_result['tls']
        features['tls_period'] = tls['period']
        features['tls_t0'] = tls['t0']
        features['tls_duration_hours'] = tls['duration'] * 24
        features['tls_depth_ppm'] = tls['depth'] * 1e6
        features['tls_sde'] = tls['snr']
        features['tls_duration_phase'] = tls['duration'] / tls['period'] if tls['period'] > 0 else np.nan
    
    # BLS vs TLS 比較特徵
    if 'bls' in search_result and 'tls' in search_result:
        bls = search_result['bls']
        tls = search_result['tls']
        
        features['period_ratio'] = tls['period'] / bls['period'] if bls['period'] > 0 else np.nan
        features['period_diff_pct'] = abs(tls['period'] - bls['period']) / bls['period'] * 100 if bls['period'] > 0 else np.nan
        features['depth_ratio'] = tls['depth'] / bls['depth'] if bls['depth'] > 0 else np.nan
        features['depth_diff_pct'] = abs(tls['depth'] - bls['depth']) / bls['depth'] * 100 if bls['depth'] > 0 else np.nan
        features['snr_ratio'] = tls['snr'] / bls['snr'] if bls['snr'] > 0 else np.nan
        features['snr_improvement'] = (tls['snr'] - bls['snr']) / bls['snr'] * 100 if bls['snr'] > 0 else np.nan
    
    # Phase 5: 去趨勢方法比較特徵
    if detrending_result:
        methods = detrending_result['methods']
        features['best_detrend_method'] = detrending_result['best_method']
        features['best_detrend_snr'] = detrending_result['best_snr']
        
        # 各方法的 SNR
        for method_key in ['lightkurve_flatten', 'wotan_biweight', 'wotan_rspline', 'wotan_hspline']:
            if method_key in methods:
                features[f'{method_key}_snr'] = methods[method_key]['snr']
        
        # SNR 改善
        if 'lightkurve_flatten' in methods and detrending_result['best_method'] != 'lightkurve_flatten':
            baseline_snr = methods['lightkurve_flatten']['snr']
            best_snr = detrending_result['best_snr']
            if baseline_snr > 0:
                features['snr_improvement_by_wotan'] = (best_snr - baseline_snr) / baseline_snr * 100
    
    # Phase 6: 奇偶深度與形狀特徵
    if 'bls' in search_result and 'lc_flat' in search_result:
        bls = search_result['bls']
        lc_flat = search_result['lc_flat']
        
        # 計算奇偶深度
        odd_even = calculate_odd_even_depth(lc_flat, bls['period'], bls['t0'], bls['duration'])
        features.update(odd_even)
        
        # 計算形狀指標
        shape = calculate_transit_shape_metrics(lc_flat, bls['period'], bls['t0'], bls['duration'])
        features.update(shape)
    
    return features

# 提取所有目標的增強特徵
print("\n開始提取增強 BLS 特徵...")
enhanced_features_list = []

for target_id in search_results.keys():
    print(f"\n🎯 提取 {search_results[target_id]['target']['name']} 的增強特徵...")
    
    # 獲取去趨勢結果
    detrend_result = detrending_results.get(target_id, None)
    
    # 提取特徵
    enhanced_features = extract_enhanced_bls_features(
        search_results[target_id],
        detrend_result
    )
    
    enhanced_features_list.append(enhanced_features)
    
    print(f"   ✅ 特徵提取完成")

# 轉換為 DataFrame
enhanced_features_df = pd.DataFrame(enhanced_features_list)

print("\n" + "="*60)
print("📊 增強特徵統計：")
print(f"   樣本數: {len(enhanced_features_df)}")
print(f"   特徵數: {len(enhanced_features_df.columns)}")
print(f"   正樣本: {(enhanced_features_df['label'] == 1).sum()}")
print(f"   負樣本: {(enhanced_features_df['label'] == 0).sum()}")

print("\n📝 新增特徵列表：")
new_features = [col for col in enhanced_features_df.columns if col not in features_df.columns]
for i, col in enumerate(new_features, 1):
    print(f"   {i}. {col}")
# ===== CELL 42 =====
# 視覺化：4種去趨勢方法的並排比較
def plot_detrending_comparison(detrending_result: Dict[str, Any]):
    """
    繪製4種去趨勢方法的並排比較圖
    """
    target = detrending_result['target']
    methods = detrending_result['methods']
    best_method = detrending_result['best_method']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"{target['name']} ({target['id']}) - 去趨勢方法比較", 
                 fontsize=14, fontweight='bold')
    
    method_names = ['lightkurve_flatten', 'wotan_biweight', 'wotan_rspline', 'wotan_hspline']
    method_titles = [
        'Lightkurve flatten()',
        'Wotan Biweight',
        'Wotan R-Spline',
        'Wotan H-Spline'
    ]
    
    for idx, (method_key, title) in enumerate(zip(method_names, method_titles)):
        ax = axes[idx // 2, idx % 2]
        
        if method_key in methods:
            lc = methods[method_key]['lc']
            snr = methods[method_key]['snr']
            
            # 繪製光曲線
            lc.scatter(ax=ax, s=0.5, color='blue', alpha=0.4)
            
            # 標題（最佳方法加星號）
            is_best = (method_key == best_method)
            title_text = f"{title}\nSNR: {snr:.2f}"
            if is_best:
                title_text = f"🏆 {title_text} 🏆"
                ax.set_facecolor('#ffffcc')  # 淡黃色背景
            
            ax.set_title(title_text, fontsize=11, fontweight='bold' if is_best else 'normal')
            ax.set_xlabel('時間 (BTJD)', fontsize=9)
            ax.set_ylabel('標準化流量', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # 計算並顯示統計資訊
            flux = lc.flux.value if hasattr(lc.flux, 'value') else np.array(lc.flux)
            flux_clean = flux[~np.isnan(flux)]
            
            textstr = f'Mean: {np.mean(flux_clean):.4f}\nStd: {np.std(flux_clean):.4f}\nPoints: {len(flux_clean):,}'
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        else:
            ax.text(0.5, 0.5, f'{title}\n資料不可用', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    return fig

# 繪製所有目標的去趨勢方法比較圖
print("\n📊 繪製去趨勢方法比較圖...")
print("="*60)

for target_id, result in detrending_results.items():
    print(f"\n📊 {result['target']['name']} - 最佳方法: {result['best_method']}")
    fig = plot_detrending_comparison(result)
# ===== CELL 43 =====
# 對每個目標執行多方法去趨勢比較
for target_id, data in processed_data.items():
    print(f"\n🎯 分析 {data['target']['name']} ({target_id})...")
    
    lc_clean = data['lc_clean']
    lc_flat_original = data['lc_flat']
    
    # 儲存各方法結果
    methods_results = {}
    
    # 1. Lightkurve flatten() - 已有的結果
    snr_lightkurve = calculate_snr(lc_flat_original)
    methods_results['lightkurve_flatten'] = {
        'lc': lc_flat_original,
        'snr': snr_lightkurve,
        'method': 'lightkurve_flatten'
    }
    print(f"   ✅ Lightkurve flatten() - SNR: {snr_lightkurve:.2f}")
    
    # 2. Wotan biweight
    lc_biweight, snr_biweight, meta_biweight = apply_wotan_detrending(
        lc_clean, method='biweight', window_length=0.5
    )
    methods_results['wotan_biweight'] = {
        'lc': lc_biweight,
        'snr': snr_biweight,
        'method': 'wotan_biweight',
        'metadata': meta_biweight
    }
    
    # 3. Wotan rspline
    lc_rspline, snr_rspline, meta_rspline = apply_wotan_detrending(
        lc_clean, method='rspline', window_length=0.5
    )
    methods_results['wotan_rspline'] = {
        'lc': lc_rspline,
        'snr': snr_rspline,
        'method': 'wotan_rspline',
        'metadata': meta_rspline
    }
    
    # 4. Wotan hspline
    lc_hspline, snr_hspline, meta_hspline = apply_wotan_detrending(
        lc_clean, method='hspline', window_length=0.5
    )
    methods_results['wotan_hspline'] = {
        'lc': lc_hspline,
        'snr': snr_hspline,
        'method': 'wotan_hspline',
        'metadata': meta_hspline
    }
    
    # 找出最佳 SNR 的方法
    best_method = max(methods_results.items(), key=lambda x: x[1]['snr'])
    best_method_name = best_method[0]
    best_snr = best_method[1]['snr']
    
    print(f"\n   🏆 最佳方法: {best_method_name} (SNR: {best_snr:.2f})")
    
    # 儲存結果
    detrending_results[target_id] = {
        'target': data['target'],
        'methods': methods_results,
        'best_method': best_method_name,
        'best_snr': best_snr
    }

print("\n✅ 所有目標的去趨勢方法比較完成！")
# ===== CELL 44 =====
# 🌟 Phase 5: Wotan Detrending Comparison
"""
比較不同去趨勢方法的效能
- Lightkurve flatten() (已使用)
- Wotan biweight
- Wotan rspline
- Wotan hspline
"""

print("="*60)
print("🌟 Phase 5: Wotan Detrending Method Comparison")
print("="*60)

# 導入 wotan
try:
    from wotan import flatten as wotan_flatten
    print("✅ Wotan 導入成功")
except ImportError:
    print("❌ Wotan 未安裝，正在安裝...")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "wotan"])
    from wotan import flatten as wotan_flatten
    print("✅ Wotan 安裝並導入成功")

def calculate_snr(lc: lk.LightCurve) -> float:
    """
    計算光曲線的信噪比 (SNR)
    
    Parameters:
    -----------
    lc : lightkurve.LightCurve
        輸入光曲線
    
    Returns:
    --------
    float : 信噪比
    """
    flux = lc.flux.value if hasattr(lc.flux, 'value') else np.array(lc.flux)
    
    # 移除 NaN 值
    flux_clean = flux[~np.isnan(flux)]
    
    if len(flux_clean) == 0:
        return 0.0
    
    # SNR = mean / std
    mean_flux = np.mean(flux_clean)
    std_flux = np.std(flux_clean)
    
    if std_flux == 0:
        return 0.0
    
    return mean_flux / std_flux

def apply_wotan_detrending(
    lc_clean: lk.LightCurve,
    method: str = 'biweight',
    window_length: float = 0.5
) -> Tuple[lk.LightCurve, float, Dict[str, Any]]:
    """
    使用 Wotan 進行去趨勢處理
    
    Parameters:
    -----------
    lc_clean : lightkurve.LightCurve
        清理過的光曲線
    method : str
        Wotan 方法: 'biweight', 'rspline', 'hspline'
    window_length : float
        滑動視窗長度（天）
    
    Returns:
    --------
    tuple : (去趨勢光曲線, SNR, metadata)
    """
    print(f"   🔧 正在使用 Wotan {method} 方法去趨勢...")
    
    start_time = time.time()
    
    # 準備資料
    time_array = lc_clean.time.value if hasattr(lc_clean.time, 'value') else np.array(lc_clean.time)
    flux_array = lc_clean.flux.value if hasattr(lc_clean.flux, 'value') else np.array(lc_clean.flux)
    
    try:
        # 執行 Wotan 去趨勢
        flatten_flux, trend_flux = wotan_flatten(
            time_array,
            flux_array,
            method=method,
            window_length=window_length,
            return_trend=True
        )
        
        # 創建新的 LightCurve 物件
        lc_wotan = lc_clean.copy()
        lc_wotan.flux = flatten_flux
        
        # 計算 SNR
        snr = calculate_snr(lc_wotan)
        
        elapsed_time = time.time() - start_time
        
        metadata = {
            'method': method,
            'window_length': window_length,
            'snr': snr,
            'elapsed_time': elapsed_time,
            'n_points': len(flatten_flux)
        }
        
        print(f"   ✅ Wotan {method} 完成（耗時 {elapsed_time:.2f} 秒）")
        print(f"      SNR: {snr:.2f}")
        
        return lc_wotan, snr, metadata
        
    except Exception as e:
        print(f"   ❌ Wotan {method} 失敗: {e}")
        # 返回原始光曲線作為 fallback
        return lc_clean, 0.0, {'method': method, 'error': str(e)}

# 儲存所有去趨勢結果
detrending_results = {}

print("\n開始對所有目標進行多方法去趨勢比較...")
print("="*60)