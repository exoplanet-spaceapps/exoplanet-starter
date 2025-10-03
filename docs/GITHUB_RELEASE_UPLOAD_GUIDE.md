# GitHub Release 資料上傳指南

**目標**: 將下載的光曲線資料（~50 GB）切分並上傳到 GitHub Releases
**限制**: 單一資產 ≤ 2 GiB，每個 release 最多 1000 個資產
**優勢**: 不計入 LFS 或 repo 容量，總容量與頻寬不設上限

---

## 📋 執行流程總覽

```
[1] 執行下載 → [2] 切分資料 → [3] 創建 Release → [4] 上傳資產 → [5] 驗證
↓
[6] 下載資料 → [7] 提取特徵 → [8] 訓練模型 → [9] 評估與部署
```

**一條龍完整流程**：從下載到模型訓練，完全自動化

---

## 1️⃣ 執行全量下載

### 步驟 1.1: Clone 並更新倉庫

```bash
git clone https://github.com/exoplanet-spaceapps/exoplanet-starter.git
cd exoplanet-starter
git pull origin main
```

### 步驟 1.2: 安裝依賴

```bash
pip install lightkurve h5py pandas numpy tqdm pyarrow
```

### 步驟 1.3: 執行下載腳本

```bash
python scripts/run_test_fixed.py
```

**預估時間**: 6-7 小時
**預估成功**: ~6,800 個樣本（57% 成功率）
**輸出目錄**: `data/lightcurves/` (約 5-10 GB)

**注意**:
- 使用 checkpoint 系統，可隨時中斷並恢復
- 下載進度保存在 `checkpoints/download_progress.parquet`

---

## 2️⃣ 切分資料（≤ 2 GiB）

### 步驟 2.1: 創建切分腳本

創建檔案 `scripts/split_for_release.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Split downloaded lightcurves into ≤ 2 GiB archives for GitHub Releases"""

import os
import tarfile
from pathlib import Path
from datetime import datetime

# 配置
PROJECT_ROOT = Path(__file__).parent.parent
LIGHTCURVE_DIR = PROJECT_ROOT / 'data' / 'lightcurves'
OUTPUT_DIR = PROJECT_ROOT / 'releases'
MAX_SIZE_GB = 1.9  # 留點空間，安全起見用 1.9 GiB
MAX_SIZE_BYTES = int(MAX_SIZE_GB * 1024 * 1024 * 1024)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("Splitting Lightcurves for GitHub Releases")
print("="*70)

# 獲取所有 HDF5 檔案
h5_files = sorted(LIGHTCURVE_DIR.glob('*.h5'))
print(f"\n[1/3] Found {len(h5_files)} HDF5 files")

if len(h5_files) == 0:
    print("  ERROR: No HDF5 files found in data/lightcurves/")
    exit(1)

# 計算總大小
total_size = sum(f.stat().st_size for f in h5_files)
print(f"  Total size: {total_size / 1024 / 1024 / 1024:.2f} GB")
print(f"  Max archive size: {MAX_SIZE_GB} GB")

# 估算需要的檔案數
estimated_archives = int(total_size / MAX_SIZE_BYTES) + 1
print(f"  Estimated archives: {estimated_archives}")

# 開始切分
print(f"\n[2/3] Creating archives...")
archive_num = 1
current_size = 0
current_files = []
created_archives = []

for h5_file in h5_files:
    file_size = h5_file.stat().st_size

    # 檢查是否需要創建新的 archive
    if current_size + file_size > MAX_SIZE_BYTES and len(current_files) > 0:
        # 創建當前 archive
        archive_name = f"lightcurves_part{archive_num:03d}.tar.gz"
        archive_path = OUTPUT_DIR / archive_name

        print(f"  Creating {archive_name} ({len(current_files)} files, {current_size/1024/1024/1024:.2f} GB)...")

        with tarfile.open(archive_path, 'w:gz') as tar:
            for f in current_files:
                arcname = f.relative_to(LIGHTCURVE_DIR)
                tar.add(f, arcname=arcname)

        created_archives.append({
            'name': archive_name,
            'path': archive_path,
            'files': len(current_files),
            'size': current_size
        })

        # 重置
        archive_num += 1
        current_files = []
        current_size = 0

    current_files.append(h5_file)
    current_size += file_size

# 處理最後一批檔案
if len(current_files) > 0:
    archive_name = f"lightcurves_part{archive_num:03d}.tar.gz"
    archive_path = OUTPUT_DIR / archive_name

    print(f"  Creating {archive_name} ({len(current_files)} files, {current_size/1024/1024/1024:.2f} GB)...")

    with tarfile.open(archive_path, 'w:gz') as tar:
        for f in current_files:
            arcname = f.relative_to(LIGHTCURVE_DIR)
            tar.add(f, arcname=arcname)

    created_archives.append({
        'name': archive_name,
        'path': archive_path,
        'files': len(current_files),
        'size': current_size
    })

# 統計
print(f"\n[3/3] Summary")
print(f"  Created {len(created_archives)} archives:")
for i, archive in enumerate(created_archives, 1):
    size_gb = archive['size'] / 1024 / 1024 / 1024
    archive_size = archive['path'].stat().st_size / 1024 / 1024 / 1024
    print(f"    {i}. {archive['name']}: {archive['files']} files, {archive_size:.2f} GB compressed")

total_compressed = sum(a['path'].stat().st_size for a in created_archives) / 1024 / 1024 / 1024
print(f"\n  Total compressed size: {total_compressed:.2f} GB")
print(f"  Compression ratio: {(1 - total_compressed / (total_size/1024/1024/1024)) * 100:.1f}%")
print(f"\n  Output directory: {OUTPUT_DIR}")

# 創建清單檔案
manifest_path = OUTPUT_DIR / 'manifest.txt'
with open(manifest_path, 'w') as f:
    f.write(f"# Exoplanet Lightcurves Archive Manifest\n")
    f.write(f"# Generated: {datetime.now().isoformat()}\n")
    f.write(f"# Total archives: {len(created_archives)}\n")
    f.write(f"# Total size: {total_compressed:.2f} GB\n\n")

    for archive in created_archives:
        size_gb = archive['path'].stat().st_size / 1024 / 1024 / 1024
        f.write(f"{archive['name']}\t{archive['files']} files\t{size_gb:.2f} GB\n")

print(f"  Manifest saved: {manifest_path}")

print("="*70)
print("✅ Archive creation complete!")
print("\nNext steps:")
print("  1. Create GitHub Release: gh release create v1.0-lightcurves --title 'Lightcurve Data v1.0'")
print("  2. Upload archives: gh release upload v1.0-lightcurves releases/*.tar.gz")
print("="*70)
```

### 步驟 2.2: 執行切分

```bash
python scripts/split_for_release.py
```

**輸出**: `releases/lightcurves_part001.tar.gz`, `part002.tar.gz`, ...
**檔案數**: 約 3-6 個（取決於實際下載大小）

---

## 3️⃣ 創建 GitHub Release

### 步驟 3.1: 安裝 GitHub CLI（如未安裝）

**Windows**:
```bash
winget install GitHub.cli
```

**macOS**:
```bash
brew install gh
```

**Linux**:
```bash
sudo apt install gh
```

### 步驟 3.2: 登入 GitHub

```bash
gh auth login
```

選擇：
- GitHub.com
- HTTPS
- Login with a web browser

### 步驟 3.3: 創建 Release

```bash
gh release create v1.0-lightcurves \
  --title "TESS Lightcurve Data v1.0" \
  --notes "Complete dataset of TESS light curves for exoplanet detection.

**Contents:**
- ~6,800 TESS light curve files (HDF5 format)
- Downloaded from MAST Archive
- Total size: ~50 GB (compressed)
- Split into parts of ≤ 2 GiB for upload

**Usage:**
See [Download Guide](https://github.com/exoplanet-spaceapps/exoplanet-starter/blob/main/docs/DOWNLOAD_FROM_RELEASE.md) for instructions on downloading and extracting.

**Dataset Info:**
- Source: NASA TESS Mission
- Targets: TOI candidates + False Positives
- Format: HDF5 (multi-sector support)
- Features: time, flux, flux_err per sector"
```

---

## 4️⃣ 上傳資產到 Release

### 方式 A: 批次上傳所有檔案（推薦）

```bash
gh release upload v1.0-lightcurves releases/*.tar.gz --clobber
```

### 方式 B: 逐一上傳（有進度顯示）

```bash
# 使用 PowerShell（Windows）
Get-ChildItem releases\*.tar.gz | ForEach-Object {
    Write-Host "Uploading $($_.Name)..."
    gh release upload v1.0-lightcurves $_.FullName
}
```

```bash
# 使用 Bash（Linux/macOS）
for file in releases/*.tar.gz; do
    echo "Uploading $(basename $file)..."
    gh release upload v1.0-lightcurves "$file"
done
```

### 方式 C: 使用 Python 腳本上傳（帶進度條）

創建 `scripts/upload_to_release.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Upload archives to GitHub Release with progress tracking"""

import subprocess
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
RELEASES_DIR = PROJECT_ROOT / 'releases'
RELEASE_TAG = 'v1.0-lightcurves'

archives = sorted(RELEASES_DIR.glob('*.tar.gz'))

print(f"Found {len(archives)} archives to upload")
print(f"Target release: {RELEASE_TAG}\n")

for archive in tqdm(archives, desc="Uploading"):
    cmd = ['gh', 'release', 'upload', RELEASE_TAG, str(archive), '--clobber']
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\n❌ Failed to upload {archive.name}")
        print(f"   Error: {result.stderr}")
    else:
        tqdm.write(f"✅ {archive.name}")

print("\n✅ Upload complete!")
```

執行：
```bash
python scripts/upload_to_release.py
```

---

## 5️⃣ 驗證上傳

### 步驟 5.1: 檢查 Release 資產

```bash
gh release view v1.0-lightcurves
```

### 步驟 5.2: 驗證檔案完整性

創建 `scripts/verify_release.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify uploaded releases match local files"""

import subprocess
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RELEASES_DIR = PROJECT_ROOT / 'releases'
RELEASE_TAG = 'v1.0-lightcurves'

# 獲取 release 資訊
cmd = ['gh', 'release', 'view', RELEASE_TAG, '--json', 'assets']
result = subprocess.run(cmd, capture_output=True, text=True)
release_data = json.loads(result.stdout)

remote_assets = {asset['name']: asset['size'] for asset in release_data['assets']}
local_files = {f.name: f.stat().st_size for f in RELEASES_DIR.glob('*.tar.gz')}

print("="*70)
print("Verification Report")
print("="*70)

print(f"\nLocal files: {len(local_files)}")
print(f"Remote assets: {len(remote_assets)}")

missing_remote = set(local_files.keys()) - set(remote_assets.keys())
missing_local = set(remote_assets.keys()) - set(local_files.keys())

if missing_remote:
    print(f"\n❌ Missing from remote: {missing_remote}")
else:
    print("\n✅ All local files uploaded")

if missing_local:
    print(f"\n⚠️ Extra files on remote: {missing_local}")

# 檢查檔案大小
size_mismatches = []
for name in set(local_files.keys()) & set(remote_assets.keys()):
    if local_files[name] != remote_assets[name]:
        size_mismatches.append(name)
        print(f"\n⚠️ Size mismatch: {name}")
        print(f"   Local: {local_files[name]:,} bytes")
        print(f"   Remote: {remote_assets[name]:,} bytes")

if not size_mismatches and not missing_remote:
    print("\n✅ All files verified successfully!")
else:
    print(f"\n⚠️ Found {len(size_mismatches)} size mismatches")

print("="*70)
```

執行：
```bash
python scripts/verify_release.py
```

---

## 6️⃣ 下載資料（其他裝置使用）

### 快速下載腳本

創建 `scripts/download_from_release.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Download and extract lightcurves from GitHub Release"""

import subprocess
import tarfile
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
LIGHTCURVE_DIR = PROJECT_ROOT / 'data' / 'lightcurves'
RELEASE_TAG = 'v1.0-lightcurves'

LIGHTCURVE_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("Downloading Lightcurves from GitHub Release")
print("="*70)

# 獲取 release 資產列表
cmd = ['gh', 'release', 'view', RELEASE_TAG, '--json', 'assets', '-q', '.assets[].name']
result = subprocess.run(cmd, capture_output=True, text=True)
assets = [name for name in result.stdout.strip().split('\n') if name.endswith('.tar.gz')]

print(f"\nFound {len(assets)} archives")

# 下載並解壓
for asset in tqdm(assets, desc="Processing"):
    print(f"\n[{asset}]")

    # 下載
    print("  Downloading...")
    cmd = ['gh', 'release', 'download', RELEASE_TAG, '-p', asset]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

    # 解壓
    print("  Extracting...")
    archive_path = PROJECT_ROOT / asset
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=LIGHTCURVE_DIR)

    # 刪除壓縮檔
    archive_path.unlink()
    print("  ✅ Done")

print("\n" + "="*70)
print("✅ Download complete!")
print(f"Files extracted to: {LIGHTCURVE_DIR}")
print("="*70)
```

使用：
```bash
gh auth login  # 首次需要登入
python scripts/download_from_release.py
```

---

## 7️⃣ 清理本地檔案（可選）

上傳完成後，可刪除本地的壓縮檔以節省空間：

```bash
# 僅刪除壓縮檔，保留原始資料
rm -rf releases/

# 或完全清理（下載的光曲線也刪除）
rm -rf data/lightcurves/
rm -rf releases/
```

---

## 📋 完整工作流程總結

```bash
# === 在下載裝置執行 ===

# 1. Clone 倉庫
git clone https://github.com/exoplanet-spaceapps/exoplanet-starter.git
cd exoplanet-starter

# 2. 下載資料（6-7 小時）
python scripts/run_test_fixed.py

# 3. 切分資料
python scripts/split_for_release.py

# 4. 創建 Release
gh release create v1.0-lightcurves --title "TESS Lightcurve Data v1.0" \
  --notes "Complete TESS light curve dataset (~50 GB)"

# 5. 上傳資產
python scripts/upload_to_release.py  # 或使用 gh release upload

# 6. 驗證
python scripts/verify_release.py

# === 在其他裝置使用 ===

# 下載資料
git clone https://github.com/exoplanet-spaceapps/exoplanet-starter.git
cd exoplanet-starter
python scripts/download_from_release.py
```

---

## 🔧 故障排除

### 問題 1: gh 命令找不到

```bash
# 確認安裝
gh --version

# 重新登入
gh auth logout
gh auth login
```

### 問題 2: 上傳超時

```bash
# 單獨重試失敗的檔案
gh release upload v1.0-lightcurves releases/lightcurves_part001.tar.gz --clobber
```

### 問題 3: 資產已存在

```bash
# 使用 --clobber 覆蓋
gh release upload v1.0-lightcurves releases/*.tar.gz --clobber
```

### 問題 4: Release 權限錯誤

確認你是 repo 的 collaborator 或 owner，並且 token 有 `repo` 權限。

---

## 📊 容量規劃

| 項目 | 大小 | 說明 |
|------|------|------|
| 原始下載 | ~10 GB | HDF5 檔案未壓縮 |
| 壓縮後 | ~5-8 GB | tar.gz 壓縮 |
| 單一資產 | ≤ 2 GB | GitHub 限制 |
| 預估檔案數 | 3-6 個 | 取決於實際大小 |
| GitHub 限制 | 1000 資產 | 足夠使用 |

---

## ✅ 檢查清單

下載裝置：
- [ ] 執行 `run_test_fixed.py` 完成下載
- [ ] 執行 `split_for_release.py` 切分資料
- [ ] 創建 GitHub Release
- [ ] 上傳所有資產
- [ ] 驗證上傳完整性
- [ ] （可選）刪除本地壓縮檔

其他裝置：
- [ ] Clone 倉庫
- [ ] 執行 `download_from_release.py`
- [ ] 驗證檔案完整性
- [ ] 繼續後續分析流程

---

## 8️⃣ 提取特徵（BLS Feature Extraction）

下載資料後，接下來提取用於機器學習的特徵。

### 步驟 8.1: 安裝額外依賴

```bash
pip install astropy scipy scikit-learn
```

### 步驟 8.2: 執行特徵提取

```bash
python scripts/test_features.py
```

**預估時間**: 15-30 分鐘（6,800 樣本）
**輸出**: `data/features.csv`（14 個特徵 × 6,800 樣本）

### 步驟 8.3: 查看特徵統計

創建 `scripts/view_features.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""View extracted features statistics"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_PATH = PROJECT_ROOT / 'data' / 'test_features.csv'

if not FEATURES_PATH.exists():
    print(f"❌ Features file not found: {FEATURES_PATH}")
    print("Run: python scripts/test_features.py")
    exit(1)

df = pd.read_csv(FEATURES_PATH)

print("="*70)
print("Feature Extraction Summary")
print("="*70)

print(f"\nTotal samples: {len(df)}")
print(f"Total features: {len(df.columns)}")

# 狀態統計
if 'status' in df.columns:
    print(f"\nExtraction status:")
    for status, count in df['status'].value_counts().items():
        print(f"  {status}: {count}")

# 標籤分布
if 'label' in df.columns:
    print(f"\nLabel distribution:")
    print(f"  Positive (exoplanet): {df['label'].sum()}")
    print(f"  Negative (no planet): {(~df['label'].astype(bool)).sum()}")

# BLS 特徵統計
successful = df[df['status'] == 'success']
if len(successful) > 0:
    print(f"\nBLS Features (successful extractions):")
    bls_cols = ['bls_period', 'bls_duration', 'bls_depth', 'bls_power', 'bls_snr']
    for col in bls_cols:
        if col in successful.columns:
            print(f"  {col}:")
            print(f"    Mean: {successful[col].mean():.4f}")
            print(f"    Std:  {successful[col].std():.4f}")
            print(f"    Min:  {successful[col].min():.4f}")
            print(f"    Max:  {successful[col].max():.4f}")

# 檢查缺失值
print(f"\nMissing values:")
null_counts = df.isnull().sum()
if null_counts.sum() == 0:
    print("  ✅ No missing values")
else:
    for col, count in null_counts[null_counts > 0].items():
        print(f"  {col}: {count}")

print("="*70)
```

執行：
```bash
python scripts/view_features.py
```

---

## 9️⃣ 訓練模型（XGBoost/LightGBM）

### 步驟 9.1: 創建訓練腳本

創建 `scripts/train_model.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train exoplanet detection model using XGBoost"""

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

warnings.filterwarnings('ignore')

# 配置
PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_PATH = PROJECT_ROOT / 'data' / 'test_features.csv'
MODEL_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("Exoplanet Detection Model Training")
print("="*70)

# 載入特徵
print("\n[1/6] Loading features...")
if not FEATURES_PATH.exists():
    print(f"❌ Features file not found: {FEATURES_PATH}")
    print("Run: python scripts/test_features.py")
    exit(1)

df = pd.read_csv(FEATURES_PATH)
print(f"  Total samples: {len(df)}")

# 篩選成功提取的樣本
successful = df[df['status'] == 'success'].copy()
print(f"  Successful extractions: {len(successful)}")

if len(successful) < 100:
    print("❌ Not enough successful samples for training (minimum 100)")
    exit(1)

# 準備特徵與標籤
print("\n[2/6] Preparing features...")
feature_cols = [
    'flux_mean', 'flux_std', 'flux_median', 'flux_mad', 'flux_skew', 'flux_kurt',
    'bls_period', 'bls_duration', 'bls_depth', 'bls_power', 'bls_snr'
]

X = successful[feature_cols].values
y = successful['label'].values

print(f"  Features: {X.shape}")
print(f"  Positive samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"  Negative samples: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")

# 分割資料
print("\n[3/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train: {X_train.shape[0]} samples")
print(f"  Test:  {X_test.shape[0]} samples")

# 訓練模型（先試 XGBoost，fallback 到 RandomForest）
print("\n[4/6] Training model...")
try:
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model_type = 'XGBoost'
    print(f"  Using {model_type}")

except ImportError:
    print("  XGBoost not found, using RandomForest instead")
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model_type = 'RandomForest'

model.fit(X_train, y_train)
print("  ✅ Training complete")

# 交叉驗證
print("\n[5/6] Cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 評估
print("\n[6/6] Evaluation...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    'model_type': model_type,
    'n_samples': len(successful),
    'n_features': len(feature_cols),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'f1_score': float(f1_score(y_test, y_pred)),
    'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
    'cv_roc_auc_mean': float(cv_scores.mean()),
    'cv_roc_auc_std': float(cv_scores.std()),
    'timestamp': datetime.now().isoformat()
}

print(f"\nTest Set Performance:")
print(f"  Accuracy:  {metrics['accuracy']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall:    {metrics['recall']:.4f}")
print(f"  F1 Score:  {metrics['f1_score']:.4f}")
print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
print(f"  FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")

# Classification Report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Planet', 'Exoplanet']))

# Feature Importance
if hasattr(model, 'feature_importances_'):
    importance = model.feature_importances_
    feature_importance = sorted(
        zip(feature_cols, importance),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nFeature Importance:")
    for feat, imp in feature_importance[:5]:
        print(f"  {feat:20s}: {imp:.4f}")

    metrics['feature_importance'] = {
        feat: float(imp) for feat, imp in feature_importance
    }

# 保存模型
model_path = MODEL_DIR / f'exoplanet_model_{model_type.lower()}.pkl'
joblib.dump(model, model_path)
print(f"\n✅ Model saved: {model_path}")

# 保存評估結果
results_path = RESULTS_DIR / 'training_results.json'
with open(results_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Results saved: {results_path}")

# 保存特徵名稱
features_meta_path = MODEL_DIR / 'feature_names.json'
with open(features_meta_path, 'w') as f:
    json.dump({'features': feature_cols}, f, indent=2)
print(f"✅ Feature metadata saved: {features_meta_path}")

print("="*70)

if metrics['roc_auc'] >= 0.80:
    print("🎉 Model training successful! (ROC-AUC ≥ 0.80)")
    print("\nNext steps:")
    print("  1. Review results: cat results/training_results.json")
    print("  2. Test inference: python scripts/predict.py")
    print("  3. Deploy to production")
else:
    print(f"⚠️ Model performance below target (ROC-AUC = {metrics['roc_auc']:.4f})")
    print("\nSuggestions:")
    print("  1. Collect more training data")
    print("  2. Feature engineering (add more BLS parameters)")
    print("  3. Hyperparameter tuning")

print("="*70)
```

### 步驟 9.2: 安裝訓練依賴

```bash
pip install xgboost scikit-learn
# 或使用 LightGBM: pip install lightgbm scikit-learn
```

### 步驟 9.3: 執行訓練

```bash
python scripts/train_model.py
```

**預估時間**: 5-15 分鐘
**輸出**:
- `models/exoplanet_model_xgboost.pkl` - 訓練好的模型
- `results/training_results.json` - 評估指標
- `models/feature_names.json` - 特徵元資料

### 步驟 9.4: 創建推論腳本

創建 `scripts/predict.py`:

```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inference script for exoplanet detection"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import json

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / 'models' / 'exoplanet_model_xgboost.pkl'
FEATURES_META_PATH = PROJECT_ROOT / 'models' / 'feature_names.json'

print("="*70)
print("Exoplanet Detection Inference")
print("="*70)

# 載入模型
if not MODEL_PATH.exists():
    print(f"❌ Model not found: {MODEL_PATH}")
    print("Run: python scripts/train_model.py")
    exit(1)

model = joblib.load(MODEL_PATH)
print(f"✅ Model loaded: {MODEL_PATH}")

# 載入特徵名稱
with open(FEATURES_META_PATH, 'r') as f:
    meta = json.load(f)
    feature_names = meta['features']

print(f"✅ Features: {len(feature_names)}")

# 測試推論（使用測試資料）
FEATURES_PATH = PROJECT_ROOT / 'data' / 'test_features.csv'
if FEATURES_PATH.exists():
    df = pd.read_csv(FEATURES_PATH)
    successful = df[df['status'] == 'success']

    if len(successful) > 0:
        print(f"\n[Testing on {len(successful)} samples]")

        X = successful[feature_names].values
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

        # 統計
        n_detected = y_pred.sum()
        high_confidence = (y_pred_proba > 0.8).sum()

        print(f"\nPredictions:")
        print(f"  Total samples: {len(y_pred)}")
        print(f"  Detected exoplanets: {n_detected} ({n_detected/len(y_pred)*100:.1f}%)")
        print(f"  High confidence (>0.8): {high_confidence}")

        # 顯示前 10 個高機率樣本
        top_indices = np.argsort(y_pred_proba)[-10:][::-1]
        print(f"\nTop 10 candidates:")
        for i, idx in enumerate(top_indices, 1):
            tic_id = successful.iloc[idx]['tic_id']
            prob = y_pred_proba[idx]
            print(f"  {i}. TIC{int(tic_id):10d} - Probability: {prob:.4f}")

        # 保存預測結果
        results_df = successful.copy()
        results_df['prediction'] = y_pred
        results_df['probability'] = y_pred_proba

        output_path = PROJECT_ROOT / 'results' / 'predictions.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\n✅ Predictions saved: {output_path}")

print("="*70)
```

### 步驟 9.5: 測試推論

```bash
python scripts/predict.py
```

---

## 🔄 完整工作流程總結（一條龍）

```bash
# === Phase 1: 資料準備與上傳（下載裝置）===

# 1. Clone 倉庫
git clone https://github.com/exoplanet-spaceapps/exoplanet-starter.git
cd exoplanet-starter

# 2. 下載資料（6-7 小時）
python scripts/run_test_fixed.py

# 3. 切分資料
python scripts/split_for_release.py

# 4. 創建 Release
gh release create v1.0-lightcurves --title "TESS Lightcurve Data v1.0" \
  --notes "Complete TESS light curve dataset (~50 GB)"

# 5. 上傳資產
python scripts/upload_to_release.py

# 6. 驗證
python scripts/verify_release.py

# === Phase 2: 模型訓練（任何裝置）===

# 7. 下載資料
git clone https://github.com/exoplanet-spaceapps/exoplanet-starter.git
cd exoplanet-starter
python scripts/download_from_release.py

# 8. 提取特徵（15-30 分鐘）
python scripts/test_features.py

# 9. 查看特徵統計
python scripts/view_features.py

# 10. 訓練模型（5-15 分鐘）
python scripts/train_model.py

# 11. 測試推論
python scripts/predict.py

# === Phase 3: 部署（可選）===

# 12. 上傳模型到 Release
gh release upload v1.0-lightcurves models/exoplanet_model_xgboost.pkl

# 13. 部署到 API 服務（根據需求）
# python scripts/deploy_api.py
```

---

## 📊 完整時間估算

| 階段 | 時間 | 備註 |
|------|------|------|
| 資料下載 | 6-7 小時 | 11,979 樣本，57% 成功率 |
| 資料切分 | 30-60 分鐘 | 壓縮為 tar.gz |
| Release 上傳 | 1-2 小時 | 取決於網路速度 |
| 資料下載（其他裝置）| 1-2 小時 | 從 GitHub Release |
| 特徵提取 | 15-30 分鐘 | BLS 算法 |
| 模型訓練 | 5-15 分鐘 | XGBoost/RandomForest |
| **總計** | **9-13 小時** | 完整一條龍流程 |

---

## ✅ 擴展檢查清單

**資料階段**：
- [ ] 執行 `run_test_fixed.py` 完成下載
- [ ] 執行 `split_for_release.py` 切分資料
- [ ] 創建 GitHub Release
- [ ] 上傳所有資產
- [ ] 驗證上傳完整性

**模型階段**：
- [ ] 從 Release 下載資料
- [ ] 執行 `test_features.py` 提取特徵
- [ ] 執行 `view_features.py` 檢查特徵品質
- [ ] 執行 `train_model.py` 訓練模型
- [ ] 執行 `predict.py` 測試推論
- [ ] 檢查 ROC-AUC ≥ 0.80

**部署階段**（可選）：
- [ ] 上傳模型到 Release
- [ ] 建立 API 服務
- [ ] 整合前端介面
- [ ] 撰寫使用文檔

---

**Created**: 2025-10-03
**Updated**: 2025-10-03
**Status**: Complete end-to-end pipeline
**Estimated time**: 9-13 hours (full pipeline)
