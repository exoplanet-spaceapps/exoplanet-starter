#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试下载脚本 - 下载 100 个样本测试流程
直接运行：python scripts/test_download.py
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置 Windows 控制台编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

warnings.filterwarnings('ignore')

print("=" * 70)
print("[TEST] Exoplanet Detection - Test Download Script")
print("=" * 70)
print()

# 检查依赖
print("[CHECK] Checking dependencies...")
try:
    import lightkurve as lk
    import joblib
    print(f"   [OK] lightkurve: {lk.__version__}")
    print(f"   [OK] numpy: {np.__version__}")
    print(f"   [OK] pandas: {pd.__version__}")
except ImportError as e:
    print(f"   [ERROR] Missing dependency: {e}")
    print("\nPlease install dependencies:")
    print("   pip install lightkurve pandas numpy tqdm joblib")
    sys.exit(1)

# 配置路径
BASE_DIR = PROJECT_ROOT
DATA_DIR = BASE_DIR / 'data'
LIGHTCURVE_DIR = DATA_DIR / 'lightcurves'
CHECKPOINT_DIR = BASE_DIR / 'checkpoints'

LIGHTCURVE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n[PATH] Working directory: {BASE_DIR}")
print(f"   Data: {DATA_DIR}")
print(f"   Lightcurves: {LIGHTCURVE_DIR}")
print(f"   Checkpoints: {CHECKPOINT_DIR}")

# 配置
CONFIG = {
    'max_workers': 4,
    'max_retries': 3,
    'timeout': 60,
    'batch_size': 100,
    'save_interval': 20,
    'test_samples': 100,  # 测试样本数
}

print(f"\n[CONFIG] Test configuration:")
for key, val in CONFIG.items():
    print(f"   {key}: {val}")

# 加载数据集
print(f"\n[DATA] Loading dataset...")
dataset_path = DATA_DIR / 'supervised_dataset.csv'
if not dataset_path.exists():
    print(f"[ERROR] Dataset not found: {dataset_path}")
    sys.exit(1)

samples_df = pd.read_csv(dataset_path)
print(f"   Total samples: {len(samples_df):,}")

# 测试模式：只取前 100 个
samples_df = samples_df.head(CONFIG['test_samples'])
print(f"   [TEST MODE] Processing {len(samples_df)} samples")

# 添加 ID
if 'sample_id' not in samples_df.columns:
    samples_df['sample_id'] = [f"SAMPLE_{i:06d}" for i in range(len(samples_df))]

if 'tic_id' not in samples_df.columns:
    if 'tid' in samples_df.columns:
        samples_df['tic_id'] = samples_df['tid']
    elif 'target_id' in samples_df.columns:
        samples_df['tic_id'] = samples_df['target_id']

print(f"   Positive: {samples_df['label'].sum()}")
print(f"   Negative: {(~samples_df['label'].astype(bool)).sum()}")


# 下载函数
def download_single_lightcurve(row: pd.Series, retries: int = 3) -> dict:
    """下载单个光曲线"""
    sample_id = row['sample_id']
    tic_id = int(float(row['tic_id']))

    result = {
        'sample_id': sample_id,
        'tic_id': tic_id,
        'status': 'failed',
        'file_path': None,
        'n_sectors': 0,
        'error': None
    }

    # 检查缓存
    file_path = LIGHTCURVE_DIR / f"{sample_id}_TIC{tic_id}.pkl"
    if file_path.exists():
        result['status'] = 'cached'
        result['file_path'] = str(file_path)
        return result

    # 下载
    for attempt in range(retries):
        try:
            search_result = lk.search_lightcurve(f"TIC {tic_id}", author='SPOC')

            if search_result is None or len(search_result) == 0:
                result['error'] = 'no_data_found'
                return result

            lc_collection = search_result.download_all()

            if lc_collection is None or len(lc_collection) == 0:
                result['error'] = 'download_failed'
                return result

            # 保存
            save_data = {
                'sample_id': sample_id,
                'tic_id': tic_id,
                'lc_collection': lc_collection,
                'n_sectors': len(lc_collection),
                'download_time': datetime.now().isoformat(),
                'sectors': [lc.meta.get('SECTOR', 'unknown') for lc in lc_collection]
            }

            joblib.dump(save_data, file_path)

            result['status'] = 'success'
            result['file_path'] = str(file_path)
            result['n_sectors'] = len(lc_collection)
            return result

        except Exception as e:
            result['error'] = str(e)[:100]  # 限制错误信息长度
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue

    return result


# 加载进度
def load_checkpoint():
    checkpoint_path = CHECKPOINT_DIR / 'download_progress.parquet'
    if checkpoint_path.exists():
        df = pd.read_parquet(checkpoint_path)
        print(f"   📂 加载检查点: {len(df)} 条记录")
        return df
    return pd.DataFrame()


def save_checkpoint(progress_df):
    checkpoint_path = CHECKPOINT_DIR / 'download_progress.parquet'
    progress_df.to_parquet(checkpoint_path, index=False)
    print(f"   💾 保存检查点: {len(progress_df)} 条记录")


# 主执行
print("\n" + "=" * 70)
print("🚀 开始测试下载")
print("=" * 70)

progress_df = load_checkpoint()

# 确定待下载
if len(progress_df) > 0:
    completed_ids = set(progress_df[progress_df['status'].isin(['success', 'cached'])]['sample_id'])
    remaining_samples = samples_df[~samples_df['sample_id'].isin(completed_ids)]
else:
    remaining_samples = samples_df.copy()

print(f"\n📊 下载进度:")
print(f"   总样本: {len(samples_df)}")
print(f"   已完成: {len(samples_df) - len(remaining_samples)}")
print(f"   待下载: {len(remaining_samples)}")

if len(remaining_samples) == 0:
    print("\n[OK] 所有样本已下载！")
else:
    print(f"\n[TIME]: {len(remaining_samples) * 5 / 60 / CONFIG['max_workers']:.1f} 分钟")
    print(f"   并发数: {CONFIG['max_workers']}")
    print()

    start_time = time.time()
    results = []

    # 并发下载
    with ThreadPoolExecutor(max_workers=CONFIG['max_workers']) as executor:
        future_to_row = {
            executor.submit(download_single_lightcurve, row, CONFIG['max_retries']): row
            for _, row in remaining_samples.iterrows()
        }

        with tqdm(total=len(remaining_samples), desc="📥 下载中") as pbar:
            for future in as_completed(future_to_row):
                result = future.result()
                results.append(result)
                pbar.update(1)

                # 定期保存
                if len(results) % CONFIG['save_interval'] == 0:
                    temp_df = pd.concat([progress_df, pd.DataFrame(results)], ignore_index=True)
                    save_checkpoint(temp_df)

                    success = sum(1 for r in results if r['status'] == 'success')
                    cached = sum(1 for r in results if r['status'] == 'cached')
                    failed = sum(1 for r in results if r['status'] == 'failed')

                    pbar.set_postfix({
                        'success': success,
                        'cached': cached,
                        'failed': failed
                    })

    # 最终保存
    if len(results) > 0:
        progress_df = pd.concat([progress_df, pd.DataFrame(results)], ignore_index=True)
        save_checkpoint(progress_df)

    elapsed = time.time() - start_time

    print(f"\n🎉 下载完成!")
    print(f"   总耗时: {elapsed / 60:.1f} 分钟")
    print(f"   平均速度: {elapsed / len(results):.1f} 秒/样本")

# 统计
print("\n" + "=" * 70)
print("📊 最终统计")
print("=" * 70)

status_counts = progress_df['status'].value_counts()
for status, count in status_counts.items():
    print(f"   {status}: {count}")

success_rate = (status_counts.get('success', 0) + status_counts.get('cached', 0)) / len(progress_df) * 100
print(f"\n   成功率: {success_rate:.1f}%")

# 验证文件
pkl_files = list(LIGHTCURVE_DIR.glob('*.pkl'))
print(f"   文件总数: {len(pkl_files)}")

if len(pkl_files) > 0:
    total_size = sum(f.stat().st_size for f in pkl_files) / 1024 / 1024
    print(f"   总大小: {total_size:.1f} MB")
    print(f"   平均大小: {total_size / len(pkl_files):.1f} MB/文件")

# 保存报告
report = {
    'timestamp': datetime.now().isoformat(),
    'test_mode': True,
    'test_samples': CONFIG['test_samples'],
    'total_samples': len(samples_df),
    'downloaded': int(status_counts.get('success', 0) + status_counts.get('cached', 0)),
    'failed': int(status_counts.get('failed', 0)),
    'success_rate': float(success_rate),
    'config': CONFIG,
    'storage': {
        'directory': str(LIGHTCURVE_DIR),
        'total_files': len(pkl_files),
        'total_size_mb': float(total_size) if len(pkl_files) > 0 else 0
    }
}

if progress_df[progress_df['status'] == 'failed']['error'].notna().any():
    report['errors'] = progress_df[progress_df['status'] == 'failed']['error'].value_counts().to_dict()

report_path = CHECKPOINT_DIR / 'test_download_report.json'
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n[OK] 报告已保存: {report_path}")

# 验证样本
if len(pkl_files) >= 3:
    print("\n" + "=" * 70)
    print("🔍 验证下载数据（随机抽样）")
    print("=" * 70)

    sample_files = np.random.choice(pkl_files, 3, replace=False)

    for pkl_file in sample_files:
        try:
            data = joblib.load(pkl_file)
            print(f"\n[OK] {pkl_file.name}")
            print(f"   TIC ID: {data['tic_id']}")
            print(f"   扇区数: {data['n_sectors']} {data['sectors']}")
            print(f"   下载时间: {data['download_time']}")

            lc = data['lc_collection'][0]
            print(f"   数据点: {len(lc.time):,}")
            print(f"   时间跨度: {float(lc.time[-1] - lc.time[0]):.1f} 天")
        except Exception as e:
            print(f"[ERROR] {pkl_file.name}: {e}")

print("\n" + "=" * 70)
print("[OK] 测试完成！")
print("=" * 70)

if success_rate >= 80:
    print("\n🎉 测试通过！可以继续以下步骤：")
    print("   1. 运行特征提取测试: python scripts/test_features.py")
    print("   2. 如果测试满意，修改配置进行全量下载")
else:
    print(f"\n[WARN] 成功率较低 ({success_rate:.1f}%)")
    print("   建议检查：")
    print("   - 网络连接")
    print("   - MAST 服务状态: https://mast.stsci.edu/")
    print(f"   - 错误详情: {report_path}")

print()
