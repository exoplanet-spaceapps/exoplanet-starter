"""
Universal Data Loader for 02_bls_baseline.ipynb
适用于 Google Colab 和本地环境的数据加载模块
"""

# Fix UTF-8 encoding for Windows environment
import sys
import io
if sys.platform == 'win32':
    # Reconfigure stdout/stderr to use UTF-8 encoding
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
from pathlib import Path
import subprocess
import os

def setup_data_directory():
    """
    设置数据目录，在 Colab 中自动从 GitHub 克隆
    """
    # 检查是否在 Colab 环境
    try:
        from google.colab import drive
        IN_COLAB = True
        print("🌍 环境: Google Colab")

        # 在 Colab 中，首先检查是否已克隆倉庫
        project_dir = Path('/content/exoplanet-starter')

        if not project_dir.exists():
            print("📥 正在从 GitHub 克隆专案...")
            result = subprocess.run([
                'git', 'clone',
                'https://github.com/exoplanet-spaceapps/exoplanet-starter.git',
                str(project_dir)
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ 专案克隆完成")
            else:
                print(f"⚠️ 克隆警告: {result.stderr}")
                print("尝试使用浅克隆...")
                subprocess.run([
                    'git', 'clone', '--depth', '1',
                    'https://github.com/exoplanet-spaceapps/exoplanet-starter.git',
                    str(project_dir)
                ], check=True)
                print("✅ 浅克隆完成")

        # 切换到专案目录
        os.chdir(str(project_dir))
        data_dir = project_dir / 'data'

    except ImportError:
        IN_COLAB = False
        data_dir = Path('../data')
        print("🌍 环境: 本地环境")

    print(f"📂 资料目录: {data_dir}")

    return data_dir, IN_COLAB


def load_datasets(data_dir):
    """
    载入所有数据集
    """
    # 验证数据目录是否存在
    if not data_dir.exists():
        print(f"❌ 资料目录不存在: {data_dir}")
        print("💡 请确保已执行 01_tap_download.ipynb 下载资料")
        return {}

    # 列出可用的资料文件
    csv_files = list(data_dir.glob('*.csv'))
    if not csv_files:
        print("⚠️ 资料目录为空，请先执行资料下载")
        return {}

    print(f"✅ 找到 {len(csv_files)} 个资料文件:")
    for f in csv_files:
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"   • {f.name} ({size_mb:.2f} MB)")

    # 载入数据集
    datasets = {}
    data_files = {
        'supervised_dataset': 'supervised_dataset.csv',
        'toi_positive': 'toi_positive.csv',
        'toi_negative': 'toi_negative.csv',
        'koi_false_positives': 'koi_false_positives.csv'
    }

    for name, filename in data_files.items():
        file_path = data_dir / filename
        if file_path.exists():
            try:
                datasets[name] = pd.read_csv(file_path)
                print(f"✅ 载入 {name}: {len(datasets[name])} 笔资料")
            except Exception as e:
                print(f"⚠️ 载入 {name} 失败: {e}")
        else:
            print(f"⚠️ 找不到档案: {file_path}")

    return datasets


def create_sample_targets(datasets, n_positive=3, n_negative=2):
    """
    从数据集中创建分析样本
    """
    sample_targets = pd.DataFrame()

    if 'supervised_dataset' in datasets and len(datasets['supervised_dataset']) > 0:
        # 从合并资料集中选取样本
        df = datasets['supervised_dataset']

        # 选取有完整资料的目标
        complete_data = df.dropna(subset=['period', 'depth'])

        # 按标签选取样本
        positive_samples = complete_data[complete_data['label'] == 1].head(n_positive)
        negative_samples = complete_data[complete_data['label'] == 0].head(n_negative)

        sample_targets = pd.concat([positive_samples, negative_samples], ignore_index=True)

        print(f"\n📊 选取分析样本: {len(sample_targets)} 个目标")
        print(f"   正样本: {(sample_targets['label'] == 1).sum()}")
        print(f"   负样本: {(sample_targets['label'] == 0).sum()}")

        # 显示样本资讯
        print("\n🎯 样本目标:")
        for idx, row in sample_targets.iterrows():
            target_id = row.get('target_id', f"Target_{idx}")
            period = row.get('period', 'N/A')
            depth = row.get('depth', 'N/A')
            source = row.get('source', 'Unknown')
            label = '正样本' if row['label'] == 1 else '负样本'

            period_str = f"P={period:.3f}d" if period != 'N/A' else "P=N/A"
            depth_str = f", depth={depth:.0f}ppm" if depth != 'N/A' else ""

            print(f"   • {target_id}: {label}, {period_str}{depth_str} ({source})")

    else:
        print("⚠️ 无法载入资料集，将使用预设目标")

        # 建立预设样本（如果无法载入资料）
        sample_targets = pd.DataFrame([
            {'target_id': 'TIC25155310', 'label': 1, 'period': 4.2, 'depth': 2800, 'source': 'default'},
            {'target_id': 'TIC307210830', 'label': 1, 'period': 3.4, 'depth': 1500, 'source': 'default'},
            {'target_id': 'KIC11904151', 'label': 1, 'period': 0.84, 'depth': 152, 'source': 'default'}
        ])

    print(f"\n✅ 资料载入完成，准备分析 {len(sample_targets)} 个目标")

    return sample_targets


def main():
    """
    主执行函数 - 完整的数据加载流程
    """
    print("="*60)
    print("🔧 载入已下载的资料集")
    print("从 GitHub 自动载入资料（适用于 Colab）")
    print("="*60)

    # 1. 设置数据目录
    data_dir, IN_COLAB = setup_data_directory()

    # 2. 载入数据集
    datasets = load_datasets(data_dir)

    # 3. 创建样本目标
    sample_targets = create_sample_targets(datasets)

    return sample_targets, datasets, data_dir, IN_COLAB


# 如果直接执行此脚本
if __name__ == "__main__":
    sample_targets, datasets, data_dir, IN_COLAB = main()
    print(f"\n✅ 数据加载完成！")
    print(f"   - 数据目录: {data_dir}")
    print(f"   - 环境: {'Google Colab' if IN_COLAB else '本地环境'}")
    print(f"   - 样本数: {len(sample_targets)}")