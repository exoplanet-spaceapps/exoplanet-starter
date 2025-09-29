# Space Apps 2025 — A World Away (Exoplanet AI) · Starter Kit

> 快速打造：**BLS/TLS 基線 + 輕量 ML 訓練（合成注入/TOI 監督） + 新資料一鍵推論 + 互動可視化**。  
> 針對 **NASA Space Apps 2025** 挑戰「**A World Away — Hunting for Exoplanets with AI**」。

---

## 為什麼選這個 Starter？
- **對題意**：需要「**在 NASA 開放資料上訓練**」並能「**分析新資料**」的 AI/ML 模型。  
- **48h 友善**：先跑 **BLS/TLS 基線** → 抽特徵 → 用 **LogReg/XGBoost** 訓練（合成注入或 TOI 監督）。  
- **Colab 友善**：所有 Notebook 皆可在 Google Colab 執行；大型檔案留在 Drive。

---

## 專案結構
```
spaceapps-exoplanet-claude-starter/
├─ app/
│  ├─ bls_features.py          # BLS/TLS 與特徵萃取
│  ├─ injection.py             # 合成凌日注入與資料產生
│  ├─ train.py                 # 訓練（LogReg/XGBoost）與校準
│  ├─ infer.py                 # 新資料端到端推論（TIC -> MAST -> 機率）
│  └─ utils.py                 # TAP/MAST/Lightkurve 小工具
├─ notebooks/
│  ├─ 01_tap_download.ipynb    # TAP 資料下載：TOI + Kepler EB
│  ├─ 02_bls_baseline.ipynb    # 基線：去趨勢 + BLS + 可視化
│  ├─ 03_injection_train.ipynb # 合成注入 + 監督式訓練管線
│  └─ 04_newdata_inference.ipynb # 新資料一鍵推論（輸入 TIC）
├─ data/                        # 資料目錄（由 notebooks 產生）
│  ├─ toi.csv                  # TOI 完整資料
│  ├─ kepler_eb.csv            # Kepler EB 資料
│  ├─ supervised_dataset.csv   # 合併訓練資料集
│  └─ data_provenance.json     # 資料來源文件
├─ queries/
│  ├─ pscomppars_example.sql   # Exoplanet Archive TAP 範例
│  ├─ toi_columns.md           # TOI 欄位與說明連結
│  └─ tap_howto.md             # TAP 使用小抄（同步/非同步、格式）
├─ web/
│  └─ app.py                   # （選用）Streamlit Demo 原型
├─ DATASETS.md                 # 可用資料集與連結（NASA/社群）
├─ CLAUDE.md                   # 用 Claude Code 開發的工作指引
├─ README.md                   # 本檔：快速上手與比賽交付指南
├─ requirements.txt            # 依賴（Colab 會以 notebook 安裝為主）
├─ .gitignore
└─ LICENSE
```

> 🔖 **Open in Colab**（建 repo 後，把下面 `USER/REPO` 換成你的倉庫路徑）：  
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USER/REPO/blob/main/notebooks/02_bls_baseline.ipynb)  
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USER/REPO/blob/main/notebooks/03_injection_train.ipynb)  
> [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USER/REPO/blob/main/notebooks/04_newdata_inference.ipynb)

---

## 快速開始（Colab）
1. 在 GitHub 建立私人/公開倉庫，將本專案推上去。
2. 在 Colab 開啟 `notebooks/03_injection_train.ipynb`：  
   - 第 1 格會自動安裝套件（包含 `lightkurve`, `astroquery`, `transitleastsquares`, `wotan`）。  
   - 若碰上 **NumPy 版本衝突**，筆記本已包含「自動降版至 `<2.0`」的處理。  
3. 訓練完成後，會在 `/content/model/` 產生：  
   - `ranker.joblib`（分類器）、`calibrator.joblib`（機率校準）、`feature_schema.json`（特徵順序）。
4. 開啟 `04_newdata_inference.ipynb`，輸入 TIC 直接跑推論。

---

## 重要資料來源（以 README 形式留存，完整連結見 DATASETS.md）
- NASA Exoplanet Archive：`pscomppars`（已知行星人口）、`toi`（TESS 候選/假陽性標註）、`koi`/`tce`（Kepler 管線產物）。
- MAST + Lightkurve：Kepler/TESS 光變曲線下載、BLS/TLS 搜尋、互動 `interact_bls()`。
- Kepler Eclipsing Binary Catalog：負樣本與壓力測試。

---

## 評估與提交（比賽友善）
- **指標**：PR-AUC、Precision@K、Recall@已知、FP（EB/假陽性）率、推論延遲。
- **不確定性**：Platt / Isotonic 校準 + 可靠度曲線。
- **可追溯**：Notebook 中保留 TAP 查詢、MAST 下載參數與原始來源連結。

---

## TAP/MAST 請求範例

### NASA Exoplanet Archive TAP 查詢
```sql
-- TOI 資料查詢（TESS Objects of Interest）
SELECT tid, toi, toipfx, tfopwg_disp, pl_orbper, pl_rade, pl_bmasse,
       st_tmag, ra, dec
FROM toi
WHERE tfopwg_disp IN ('PC', 'CP', 'KP', 'FP')
ORDER BY tid

-- 確認行星參數查詢
SELECT pl_name, hostname, pl_orbper, pl_rade, pl_masse,
       st_teff, st_rad, disc_year
FROM pscomppars
WHERE disc_facility = 'Transiting Exoplanet Survey Satellite (TESS)'
```

### MAST Lightkurve 下載
```python
import lightkurve as lk

# 搜尋 TESS 光曲線
search_result = lk.search_lightcurve(
    "TIC 25155310",
    mission="TESS",
    author="SPOC"
)

# 下載並處理
lc = search_result[0].download()
lc_clean = lc.remove_nans()
lc_flat = lc_clean.flatten(window_length=401)
```

## 資料版本與來源

- **NASA Exoplanet Archive**: 2025年1月版本
  - TOI 表：7000+ 候選天體
  - pscomppars：5600+ 確認行星
  - API 端點：https://exoplanetarchive.ipac.caltech.edu/TAP

- **MAST Archive**:
  - TESS 資料：Sectors 1-70
  - 處理版本：SPOC v5.0
  - API 端點：https://mast.stsci.edu/api/

- **Kepler EB Catalog**: Version 3 (2016)
  - 2877 個雙星系統
  - 來源：http://keplerebs.villanova.edu/

## 限制與風險

### 模型限制
- **偵測範圍**：最佳化於 0.5-20 天週期，深度 >500 ppm
- **資料品質**：需要至少 100 個有效資料點
- **假陽性源**：雙星系統、背景混合、儀器效應
- **訓練偏差**：合成注入可能無法完全模擬真實系統誤差

### 技術風險
- **API 依賴**：需要穩定網路連接至 NASA/MAST
- **計算資源**：批次處理需要充足記憶體（建議 >8GB）
- **版本相容**：NumPy <2.0 限制（Lightkurve 相容性）

### 使用建議
- 高信心候選（>0.8）仍需人工驗證
- 定期使用新 TESS 資料重新訓練
- 考慮多扇區觀測以提高可靠性
- 檢查已知行星資料庫避免重複發現

## 引用與致謝

使用本專案請引用：
```bibtex
@software{exoplanet_ai_2025,
  title = {Exoplanet AI Detection Pipeline},
  author = {Space Apps 2025 Team},
  year = {2025},
  url = {https://github.com/exoplanet-spaceapps/exoplanet-starter}
}
```

資料來源引用：
- NASA Exoplanet Archive: https://doi.org/10.26133/NEA12
- TESS Mission: Ricker et al. 2015, JATIS, 1, 014003
- Lightkurve: Lightkurve Collaboration, 2018

## 授權
- 程式碼：MIT License（見 LICENSE）。
- 資料：依各資料源條款（NASA/MAST/Exoplanet Archive/HLSP 等）使用與引用；在論文/專案頁逐條標註來源。
