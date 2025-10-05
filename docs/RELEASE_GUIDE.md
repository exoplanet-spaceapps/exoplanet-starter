# GitHub Release 發佈指南

## 📦 發佈 balanced_features.csv 到 GitHub Release

### 前置條件

等待 `scripts/extract_balanced_features.py` 完成後，確認以下檔案存在：
- `data/balanced_features.csv` (應該包含 1,000 筆特徵資料)

### 步驟 1: 創建 Git Tag

```bash
# 創建標籤
git tag -a v1.0-features -m "Release: Balanced features dataset (500 True + 500 False samples)"

# 推送標籤到 GitHub
git push origin v1.0-features
```

### 步驟 2: 在 GitHub 網站創建 Release

1. **前往 Releases 頁面**
   ```
   https://github.com/exoplanet-spaceapps/exoplanet-starter/releases
   ```

2. **點擊 "Draft a new release"**

3. **填寫 Release 資訊**
   - **Tag**: 選擇 `v1.0-features`
   - **Release title**: `Balanced Features Dataset v1.0`
   - **Description**:
     ```markdown
     # Exoplanet Detection - Balanced Features Dataset

     ## 📊 Dataset Information

     - **Total samples**: 1,000
     - **True samples (exoplanet)**: 500
     - **False samples (no exoplanet)**: 500
     - **Features**: 11 BLS-based features
     - **File format**: CSV
     - **File size**: ~200-300 KB

     ## 🔬 Features Included

     1. **Flux Statistics** (6 features):
        - flux_mean, flux_std, flux_median
        - flux_mad, flux_skew, flux_kurt

     2. **BLS Transit Detection** (5 features):
        - bls_period, bls_duration, bls_depth
        - bls_power, bls_snr

     ## 📥 Download & Usage

     ### Direct Download
     ```bash
     wget https://github.com/exoplanet-spaceapps/exoplanet-starter/releases/download/v1.0-features/balanced_features.csv
     ```

     ### Google Colab
     Use the notebook: `notebooks/04_Google_Colab_Training.ipynb`

     The notebook will automatically download this file.

     ### Local Training
     ```bash
     # Place file in data/ directory
     mv balanced_features.csv data/

     # Train model
     python scripts/train_model.py
     ```

     ## 🎯 Model Training

     This dataset is ready for XGBoost binary classification:
     - Train/Test split: 80/20
     - Expected accuracy: 85-95%
     - Expected ROC-AUC: 0.85-0.95

     ## 📝 Citation

     If you use this dataset, please cite:
     ```
     Exoplanet Detection using TESS Lightcurves
     NASA Space Apps Challenge 2025
     https://github.com/exoplanet-spaceapps/exoplanet-starter
     ```

     ## 🔗 Related Files

     - Source: TESS Lightcurve Data (MAST Archive)
     - Processing: `scripts/extract_balanced_features.py`
     - Training: `notebooks/04_Google_Colab_Training.ipynb`
     ```

4. **上傳檔案**
   - 點擊 "Attach binaries by dropping them here or selecting them"
   - 選擇 `data/balanced_features.csv`
   - 等待上傳完成

5. **發佈 Release**
   - 確認所有資訊正確
   - 點擊 "Publish release"

### 步驟 3: 驗證 Release

發佈後，您的下載 URL 應該是：

```
https://github.com/exoplanet-spaceapps/exoplanet-starter/releases/download/v1.0-features/balanced_features.csv
```

測試下載：

```bash
curl -L -o test_download.csv https://github.com/exoplanet-spaceapps/exoplanet-starter/releases/download/v1.0-features/balanced_features.csv

# 驗證檔案
head test_download.csv
wc -l test_download.csv  # 應該顯示 1001 行 (1 header + 1000 data)
```

### 步驟 4: 更新 Colab 筆記本（已完成）

筆記本中的設定已經正確：

```python
REPO_OWNER = "exoplanet-spaceapps"
REPO_NAME = "exoplanet-starter"
RELEASE_TAG = "v1.0-features"
ASSET_NAME = "balanced_features.csv"
```

## 🚀 快速命令

等特徵提取完成後，執行：

```bash
# 1. 創建並推送標籤
git tag -a v1.0-features -m "Release: Balanced features dataset (500+500)" && git push origin v1.0-features

# 2. 然後前往 GitHub 網站完成 Release 創建和檔案上傳
echo "Next: Visit https://github.com/exoplanet-spaceapps/exoplanet-starter/releases/new"
```

## ❓ 疑難排解

### 問題: Tag 已存在
```bash
# 刪除本地標籤
git tag -d v1.0-features

# 刪除遠端標籤
git push origin :refs/tags/v1.0-features

# 重新創建
git tag -a v1.0-features -m "Release: Balanced features dataset"
git push origin v1.0-features
```

### 問題: 檔案太大
如果 CSV 檔案 > 100 MB，考慮壓縮：

```bash
# 壓縮檔案
gzip -c data/balanced_features.csv > balanced_features.csv.gz

# 上傳 .gz 檔案到 Release
# 並更新 Colab 筆記本的 ASSET_NAME = "balanced_features.csv.gz"
```

## 📊 檔案檢查清單

上傳前確認：

- [ ] 檔案存在: `data/balanced_features.csv`
- [ ] 檔案大小合理 (< 10 MB)
- [ ] 包含 1000-1001 行 (1 header + 1000 data)
- [ ] 包含所有 11 個特徵欄位
- [ ] 包含 `label` 欄位 (0/1)
- [ ] 包含 `status` 欄位 (應該全部是 "success")

驗證命令：

```bash
wc -l data/balanced_features.csv
head -1 data/balanced_features.csv | tr ',' '\n' | wc -l  # 應該顯示 ~14-15 欄位
```
