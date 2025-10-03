# Notebooks 資料夾清理計劃

## 📋 當前檔案分析

### Notebook 檔案 (16 個)

#### 保留的核心檔案 ✅
1. **00_verify_datasets.ipynb** (18K) - 資料驗證，保留
2. **01_tap_download.ipynb** (60K) - 資料下載，保留
3. **02_bls_baseline.ipynb** (84K) - 標準 BLS 分析，保留
4. **03_injection_train_FIXED.ipynb** (84K) - ✨ 修復版本，主要使用
5. **03_injection_train_MINIMAL.ipynb** (22K) - 精簡版，保留
6. **04_newdata_inference.ipynb** (68K) - 推論，保留
7. **05_metrics_dashboard.ipynb** (151K) - 評估儀表板，保留

#### 應該刪除的過時檔案 ❌
1. **02_bls_baseline_batch.ipynb** (17K) - 批次處理版本，已被 ENHANCED 取代
2. **02_bls_baseline_COLAB.ipynb** (41K) - 舊版 Colab，已有 ENHANCED
3. **02_bls_baseline_COLAB_ENHANCED.ipynb** (47K) - 有穩定性問題，不推薦使用
4. **02_bls_baseline_LOCAL.ipynb** (88K) - 本地版本，功能重複
5. **03_injection_train.ipynb** (130K) - 舊版本，已被 FIXED 取代
6. **03_injection_train_executed.ipynb** (134K) - 已執行版本，不需要
7. **03_injection_train_MINIMAL_executed_BALANCED.ipynb** (2.2M) - 巨大檔案，已執行版本
8. **04_newdata_inference_executed.ipynb** (79K) - 已執行版本，不需要
9. **05_metrics_dashboard_executed.ipynb** (151K) - 已執行版本，不需要

#### 輔助檔案分析

##### 保留 ✅
- **data_loader_colab.py** (6.7K) - 資料載入工具，必要
- **README_MINIMAL.md** (4.5K) - 文件說明

##### 刪除 ❌
- **02_bls_baseline_COLAB_PARALLEL.py** (19K) - 測試檔案，已整合到 notebook
- **DIAGNOSIS_REPORT.md** (6.3K) - 診斷報告，過時
- **FIX_NOTEBOOK_03_IMPORTS.md** (7.7K) - 修復說明，已完成
- **github_push_cell_2025.py** (13K) - GitHub 推送工具，應在 scripts/
- **improved_github_push.py** (22K) - GitHub 推送工具，應在 scripts/
- **parallel_extraction_module.py** (5.5K) - 並行處理模組，已整合
- **QUICK_FIX_CELL.py** (4K) - 快速修復，已完成
- **quick_test.py** (1.2K) - 測試檔案
- **STATUS.md** (5.1K) - 狀態報告，過時
- **test_02_simple.py** (4.8K) - 測試檔案
- **TEST_RESULTS.md** (7.3K) - 測試結果，過時

## 🎯 清理行動

### 第一步：創建備份目錄
```bash
mkdir -p notebooks/archive/outdated_notebooks
mkdir -p notebooks/archive/old_docs
mkdir -p notebooks/archive/test_files
```

### 第二步：移動過時 Notebooks（9 個，總計 3.1 MB）
```bash
mv notebooks/02_bls_baseline_batch.ipynb notebooks/archive/outdated_notebooks/
mv notebooks/02_bls_baseline_COLAB.ipynb notebooks/archive/outdated_notebooks/
mv notebooks/02_bls_baseline_COLAB_ENHANCED.ipynb notebooks/archive/outdated_notebooks/
mv notebooks/02_bls_baseline_LOCAL.ipynb notebooks/archive/outdated_notebooks/
mv notebooks/03_injection_train.ipynb notebooks/archive/outdated_notebooks/
mv notebooks/03_injection_train_executed.ipynb notebooks/archive/outdated_notebooks/
mv notebooks/03_injection_train_MINIMAL_executed_BALANCED.ipynb notebooks/archive/outdated_notebooks/
mv notebooks/04_newdata_inference_executed.ipynb notebooks/archive/outdated_notebooks/
mv notebooks/05_metrics_dashboard_executed.ipynb notebooks/archive/outdated_notebooks/
```

### 第三步：移動輔助檔案（11 個）
```bash
# 移動文件到 archive
mv notebooks/DIAGNOSIS_REPORT.md notebooks/archive/old_docs/
mv notebooks/FIX_NOTEBOOK_03_IMPORTS.md notebooks/archive/old_docs/
mv notebooks/STATUS.md notebooks/archive/old_docs/
mv notebooks/TEST_RESULTS.md notebooks/archive/old_docs/

# 移動測試檔案到 archive
mv notebooks/02_bls_baseline_COLAB_PARALLEL.py notebooks/archive/test_files/
mv notebooks/parallel_extraction_module.py notebooks/archive/test_files/
mv notebooks/QUICK_FIX_CELL.py notebooks/archive/test_files/
mv notebooks/quick_test.py notebooks/archive/test_files/
mv notebooks/test_02_simple.py notebooks/archive/test_files/

# 移動 GitHub 工具到 scripts/
mv notebooks/github_push_cell_2025.py scripts/
mv notebooks/improved_github_push.py scripts/
```

## 📊 清理後結果

### Notebooks 目錄內容（保留 9 個檔案）

#### Notebook 檔案（7 個）
```
00_verify_datasets.ipynb              (18K)
01_tap_download.ipynb                 (60K)
02_bls_baseline.ipynb                 (84K)
03_injection_train_FIXED.ipynb        (84K) ⭐ 主要使用
03_injection_train_MINIMAL.ipynb      (22K)
04_newdata_inference.ipynb            (68K)
05_metrics_dashboard.ipynb           (151K)
```

#### 輔助檔案（2 個）
```
data_loader_colab.py                  (6.7K)
README_MINIMAL.md                     (4.5K)
```

### 釋放空間
- **移動檔案總大小**：約 3.5 MB
- **目錄變化**：從 27 個檔案減少到 9 個檔案
- **減少比例**：67% 的檔案被整理到 archive

### Archive 目錄結構
```
notebooks/archive/
├── outdated_notebooks/     (9 個 .ipynb 檔案)
├── old_docs/              (4 個 .md 檔案)
└── test_files/            (5 個 .py 檔案)
```

## ✅ 執行清理

執行以下命令進行清理：

```bash
cd C:\Users\tingy\Desktop\dev\exoplanet-starter
bash notebooks/execute_cleanup.sh
```

或手動執行上述命令。

## 🔍 驗證清理結果

```bash
# 查看保留的檔案
ls -lh notebooks/*.ipynb

# 查看 archive 內容
ls -R notebooks/archive/

# 檢查釋放的空間
du -sh notebooks/archive/
```

## 🎯 推薦的 Notebook 使用順序

執行順序（清理後）：
```
1. 00_verify_datasets.ipynb    (驗證資料)
2. 01_tap_download.ipynb       (下載資料)
3. 02_bls_baseline.ipynb       (BLS 分析，可選)
4. 03_injection_train_FIXED.ipynb ⭐ (訓練模型)
5. 04_newdata_inference.ipynb  (推論)
6. 05_metrics_dashboard.ipynb  (評估)
```

快速測試流程（跳過 02）：
```
1. 01_tap_download.ipynb
2. 03_injection_train_MINIMAL.ipynb ⚡
3. 04_newdata_inference.ipynb
```