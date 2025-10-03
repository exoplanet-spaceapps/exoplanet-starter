# GitHub Release 資料上傳指南

**目標**: 將下載的光曲線資料（~50 GB）切分並上傳到 GitHub Releases
**限制**: 單一資產 ≤ 2 GiB，每個 release 最多 1000 個資產
**優勢**: 不計入 LFS 或 repo 容量，總容量與頻寬不設上限

---

## 📋 執行流程總覽

```
[1] 執行下載 → [2] 切分資料 → [3] 創建 Release → [4] 上傳資產 → [5] 驗證
```

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

**Created**: 2025-10-03
**Status**: Ready for execution
**Estimated time**: 7-8 hours (download) + 1-2 hours (split & upload)
