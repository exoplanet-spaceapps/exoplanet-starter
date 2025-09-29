# 🧠 Exoplanet Detection Project - Memory System

**專案狀態**: 開發中 (Phase 2 完成)
**最後更新**: 2025-01-29
**下次繼續**: 執行 02_bls_baseline.ipynb 並繼續後續分析

---

## 📋 專案概述

### 目標
使用 NASA 公開資料訓練一個可分析新資料的系外行星偵測/排序器，包含：
- BLS/TLS 基線分析
- 合成注入或 TOI 監督訓練
- 機率校準
- 新資料一鍵推論

### 技術架構
- **資料來源**: NASA Exoplanet Archive (TOI, KOI False Positives)
- **分析方法**: BLS (Box Least Squares) + TLS (Transit Least Squares)
- **機器學習**: LogisticRegression, RandomForest, XGBoost + 機率校準
- **部署環境**: Google Colab (主要) + 本地環境支援

---

## ✅ **已完成階段 (Phase 1-2)**

### Phase 1: 資料下載與基礎設施 ✅
**檔案**: `01_tap_download.ipynb`
**狀態**: 完成並已推送到 GitHub

#### 主要成果:
- ✅ 下載真實 NASA TOI 資料 (2000+ 筆)
- ✅ 下載 KOI False Positives 作為負樣本 (50+ 筆)
- ✅ 建立完整的監督學習資料集 (`supervised_dataset.csv`)
- ✅ 資料品質文档化 (`data_provenance.json`)
- ✅ **解決 Git LFS 追蹤錯誤問題**
- ✅ **修復 GitHub 推送中的目錄創建問題**

#### 輸出檔案:
```
data/
├── toi.csv - 完整 TOI 資料
├── toi_positive.csv - TOI 正樣本 (PC/CP/KP)
├── toi_negative.csv - TOI 負樣本 (FP)
├── koi_false_positives.csv - KOI False Positives
├── supervised_dataset.csv - 合併訓練資料集 ⭐
└── data_provenance.json - 資料來源文檔
```

#### 關鍵技術解決方案:
1. **NumPy 2.0 相容性修復**:
   ```bash
   pip install numpy==1.26.4 scipy'<1.13'
   ```
2. **TOI 欄位映射修復**:
   - `pl_orbper` → `toi_period`
   - `pl_trandep` → `toi_depth`
   - `pl_trandurh` → `toi_duration` (小時→天轉換)

3. **GitHub Push 終極解決方案**:
   - 自動環境檢測 (Colab/本地)
   - 自動目錄創建和 Git LFS 設定
   - 智能衝突解決 (`git pull --rebase`)

### Phase 2: BLS/TLS 基線分析準備 ✅
**檔案**: `02_bls_baseline.ipynb`
**狀態**: 已檢查和優化，準備執行

#### 預期功能:
- ✅ 自動載入 01 的資料集
- ✅ 下載 TESS/Kepler 光曲線
- ✅ BLS (Box Least Squares) 週期搜尋
- ✅ TLS (Transit Least Squares) 高精度分析
- ✅ 特徵提取 → `bls_tls_features.csv`
- ✅ 完整的錯誤處理和 fallback 機制

---

## 🔧 **已解決的關鍵技術問題**

### 1. Git LFS 追蹤錯誤 ❌→✅
**問題**: `Command '['git', 'lfs', 'track', '*.csv']' returned non-zero exit status 128`
**原因**: Colab 環境中 Git 倉庫未正確初始化
**解決方案**:
```python
# 完整的 Git 倉庫初始化流程
subprocess.run(['git', 'init'], check=True)
subprocess.run(['apt-get', 'install', '-y', '-qq', 'git-lfs'], check=True)
subprocess.run(['git', 'lfs', 'install', '--skip-repo'], capture_output=True)
```

### 2. GitHub 推送目錄缺失錯誤 ❌→✅
**問題**: `❌ data 目錄不存在`
**原因**: Colab 環境中缺少專案目錄結構
**解決方案**: 增強 `ultimate_push_to_github()` 自動創建目錄
```python
essential_dirs = ['data', 'notebooks']
for dir_name in essential_dirs:
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)
```

### 3. TOI 資料欄位映射問題 ❌→✅
**問題**: NASA Archive 使用 `pl_*` 前綴而非 `toi_*`
**解決方案**: 建立完整的欄位映射表
```python
column_mapping = {
    'toi_period': 'pl_orbper',
    'toi_depth': 'pl_trandep',
    'toi_duration': 'pl_trandurh'  # 需要小時→天轉換
}
```

### 4. NumPy 2.0 天文學套件相容性 ❌→✅
**問題**: `transitleastsquares` 等套件不支援 NumPy 2.0
**解決方案**: 強制降級並提供清楚的重啟指示
```bash
pip install -q numpy==1.26.4 astropy scipy'<1.13'
```

---

## 🚀 **下一步開發計劃 (Phase 3-5)**

### Phase 3: BLS/TLS 特徵分析 📋 (即將開始)
**檔案**: `02_bls_baseline.ipynb`
**預期時間**: 1-2 小時

#### 待執行任務:
- [ ] 執行 BLS/TLS 分析 (3-5 個目標)
- [ ] 生成功率譜和摺疊光曲線圖
- [ ] 提取 ML 特徵 (`bls_tls_features.csv`)
- [ ] 推送結果到 GitHub

#### 已知風險與對策:
- **光曲線下載失敗**: 有預設目標 fallback
- **記憶體不足**: 已限制搜尋範圍和使用多線程
- **套件相容性**: 有完整的安裝和重啟流程

### Phase 4: 監督學習訓練 📋
**檔案**: `03_injection_train.ipynb`
**目標**: 訓練分類器並進行機率校準

#### 計劃功能:
- [ ] 載入真實 TOI + KOI FP 資料
- [ ] 合成注入資料生成 (可選)
- [ ] 多模型訓練 (LogReg, RF, XGBoost)
- [ ] Isotonic/Platt 機率校準
- [ ] 模型持久化 (`model/ranker.joblib`)

### Phase 5: 新資料推論 📋
**檔案**: `04_newdata_inference.ipynb`
**目標**: 一鍵推論新的 TIC 目標

#### 計劃功能:
- [ ] TIC → MAST → BLS/TLS → 機率 pipeline
- [ ] 批次處理多目標
- [ ] GPU 優化 (如果可用)
- [ ] 結果排序和可視化

### Phase 6: 評估儀表板 📋
**檔案**: `05_metrics_dashboard.ipynb`
**目標**: 全面的模型評估

#### 計劃功能:
- [ ] PR-AUC, ROC-AUC, ECE, Brier Score
- [ ] 錯誤分析和案例研究
- [ ] 效能基準測試
- [ ] 最終評估報告

---

## 🗂️ **專案文件結構**

```
exoplanet-starter/
├── README.md                     # 專案說明
├── CLAUDE.md                     # Claude 開發指引 ⭐
├── DATASETS.md                   # 資料說明文檔
├── PROJECT_MEMORY.md             # 本記憶系統檔案 ⭐
├── requirements.txt              # Python 依賴
├── data/                         # 資料目錄 (Git LFS)
│   ├── supervised_dataset.csv    # 主訓練資料集
│   ├── bls_tls_features.csv      # BLS/TLS 特徵 (待生成)
│   └── *.csv                     # 其他資料檔案
├── notebooks/                    # 分析筆記本
│   ├── 01_tap_download.ipynb     # ✅ 資料下載
│   ├── 02_bls_baseline.ipynb     # 📋 BLS/TLS 分析 (下一步)
│   ├── 03_injection_train.ipynb  # 📋 ML 訓練
│   ├── 04_newdata_inference.ipynb # 📋 新資料推論
│   └── 05_metrics_dashboard.ipynb # 📋 評估儀表板
├── app/                          # 核心程式碼模組
└── model/                        # 訓練好的模型 (待建立)
```

---

## 💡 **關鍵洞察與技術筆記**

### 資料品質洞察:
1. **TOI 資料** (NASA Archive): 高品質但物理參數缺失較多
2. **KOI False Positives**: 品質好，適合作為負樣本
3. **資料平衡**: 正負樣本比例約 7:3，可能需要 SMOTE 或類似技術

### GitHub 整合模式:
- 每個 notebook 都有獨立的 `ultimate_push_to_github_XX()` 函數
- 自動處理 Colab ↔ GitHub 同步
- 支援版本衝突自動解決

### 效能優化策略:
- **BLS**: 快速篩選，適合大規模搜尋
- **TLS**: 高精度確認，計算量大但準確
- **組合策略**: BLS 初篩 → TLS 精確分析

---

## 🔗 **重要連結與資源**

### NASA 資料源:
- [TOI Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)
- [KOI Cumulative Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
- [TAP Service](https://exoplanetarchive.ipac.caltech.edu/TAP)

### 技術文檔:
- [Lightkurve 文檔](https://docs.lightkurve.org/)
- [TransitLeastSquares 文檔](https://github.com/hippke/tls)
- [Astroquery 文檔](https://astroquery.readthedocs.io/)

### 已知問題與解決方案:
- **NumPy 2.0**: 使用 1.26.4 版本
- **Git LFS**: 需要完整的倉庫初始化
- **目錄結構**: 自動創建機制已實現

---

## 🎯 **繼續開發時的檢查清單**

### 環境準備:
- [ ] 確認 Python 環境 (推薦 Google Colab)
- [ ] 檢查 `data/supervised_dataset.csv` 是否存在
- [ ] 確認 GitHub Token 有效 (如需推送)

### 代碼執行順序:
1. **02_bls_baseline.ipynb**:
   - 執行 Cell 4 (套件安裝) → 重啟 Runtime
   - 從 Cell 6 開始執行
2. **03_injection_train.ipynb**: 載入 BLS/TLS 特徵進行訓練
3. **04_newdata_inference.ipynb**: 測試推論管線
4. **05_metrics_dashboard.ipynb**: 生成最終評估

### 故障排除:
- **套件錯誤**: 檢查 NumPy 版本並重啟 Runtime
- **資料載入失敗**: 確認 `data/` 目錄和檔案存在
- **推送失敗**: 檢查 Token 權限和網路連接

---

**🎯 下次開始**: 直接執行 `02_bls_baseline.ipynb` 即可繼續！

---
*Generated by Claude Code - Exoplanet Detection Project*
*Last Updated: 2025-01-29*