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
│  ├─ 04_newdata_inference.ipynb # 新資料一鍵推論（輸入 TIC）
│  └─ 05_metrics_dashboard.ipynb # 評估指標與模型比較儀表板
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

## 🚀 **一鍵開啟 Colab**

將 `YOUR_USERNAME/YOUR_REPO` 換成你的 GitHub 倉庫路徑：

| Notebook | 說明 | Colab 連結 |
|----------|------|-----------|
| 01_tap_download | TAP 資料下載（TOI + Kepler EB）| [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebooks/01_tap_download.ipynb) |
| 02_bls_baseline | BLS/TLS 基線分析與可視化 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebooks/02_bls_baseline.ipynb) |
| 03_injection_train | 合成注入 + 監督學習訓練 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebooks/03_injection_train.ipynb) |
| 04_newdata_inference | 新資料端到端推論 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebooks/04_newdata_inference.ipynb) |
| 05_metrics_dashboard | 評估指標與模型比較 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/notebooks/05_metrics_dashboard.ipynb) |

### 範例連結（可直接測試）
使用預設倉庫路徑的可用連結：
- [🔬 **基線分析**](https://colab.research.google.com/github/exoplanet-spaceapps/exoplanet-starter/blob/main/notebooks/02_bls_baseline.ipynb) - BLS/TLS 搜尋與可視化
- [🤖 **模型訓練**](https://colab.research.google.com/github/exoplanet-spaceapps/exoplanet-starter/blob/main/notebooks/03_injection_train.ipynb) - 合成注入與監督學習
- [🎯 **推論測試**](https://colab.research.google.com/github/exoplanet-spaceapps/exoplanet-starter/blob/main/notebooks/04_newdata_inference.ipynb) - TIC 輸入一鍵推論

---

## 快速開始（GitHub → Colab 工作流程）

### 🛠️ **建立你的倉庫**
```bash
# 1. Fork 或複製此專案到你的 GitHub
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# 2. （可選）用 Claude Code 繼續開發
claude --project-path . "help me improve the model"
```

### 📝 **開啟 Colab Notebooks**
1. **直接從 GitHub 開啟**：
   - 點擊上方表格的 [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)] 徽章
   - 或進入 [colab.research.google.com](https://colab.research.google.com) → **File** → **Open notebook** → **GitHub 分頁** → 搜索 `YOUR_USERNAME/YOUR_REPO`

2. **執行訓練**（推薦從 `03_injection_train.ipynb` 開始）：
   ```python
   # 第 1 格自動安裝所有依賴
   !pip install lightkurve astroquery transitleastsquares wotan numpy"<2.0"

   # 若有 GPU，啟用混合精度訓練（見下方 L4 最佳化）
   import torch
   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"🚀 使用裝置: {device}")
   ```

3. **模型輸出**：
   - 訓練完成後會在 `/content/model/` 產生模型檔案
   - 若要持久化，連結 Google Drive：`from google.colab import drive; drive.mount('/content/drive')`

4. **開啟推論**：
   - 點擊 `04_newdata_inference.ipynb` 徽章直接測試推論管線
   - 輸入 TIC ID（如 `TIC 25155310`）即可得到行星候選機率

### 🔄 **開發迭代循環**
```bash
# 在本機用 Claude Code 修改
claude "add feature X to the injection pipeline"

# 提交更改
git add . && git commit -m "feat: add feature X"
git push origin main

# 回到 Colab，重新開啟 notebook 即可看到最新版本
```

## 🚀 **Colab GPU 最佳化（L4 / T4 加速）**

### L4 Ada 架構 + BFloat16 混合精度
Google Colab Pro/Enterprise 提供 NVIDIA L4 GPU，支援硬體加速的 BF16 訓練：

```python
# 在訓練 notebook 中貼入以下程式碼區塊
import torch
from torch.cuda.amp import autocast, GradScaler
import numpy as np

# 檢查 GPU 型號與 BF16 支援
def setup_gpu_training():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU: {gpu_name}")

        # L4 (Ada Lovelace) 最佳化
        if "L4" in gpu_name or "Ada" in gpu_name:
            print("✅ 偵測到 L4 GPU - 啟用 BFloat16 加速")
            return True, torch.bfloat16
        else:
            print("⚡ 使用 FP16 混合精度")
            return True, torch.float16
    else:
        print("❌ CPU 模式")
        return False, torch.float32

# 混合精度訓練循環
def train_with_amp(model, train_loader, optimizer, criterion):
    use_amp, dtype = setup_gpu_training()
    scaler = GradScaler(enabled=use_amp)

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad(set_to_none=True)  # 記憶體最佳化

        if use_amp:
            with autocast(dtype=dtype):
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}, Loss: {loss.item:.4f}')

# 推論最佳化
@torch.inference_mode()  # 比 torch.no_grad() 更快
def predict_batch(model, data_loader):
    model.eval()
    predictions = []

    for batch in data_loader:
        with autocast(dtype=torch.bfloat16):  # L4 推論加速
            output = model(batch)
            predictions.append(output.cpu())

    return torch.cat(predictions)
```

### 記憶體與快取最佳化
```python
# Colab 環境資訊
!nvidia-smi
!cat /proc/meminfo | grep MemTotal
!df -h /content  # 磁碟空間

# 清理快取避免 OOM
import gc
def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU 記憶體: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# 批次處理大資料集
def process_large_lightcurves(tic_list, batch_size=32):
    for i in range(0, len(tic_list), batch_size):
        batch_tics = tic_list[i:i+batch_size]
        # 處理批次
        yield batch_tics
        cleanup_memory()  # 每批次後清理
```

### Colab 專屬技巧
```python
# 連結 Google Drive 持久化模型
from google.colab import drive
drive.mount('/content/drive')

# 建立專案目錄
import os
project_dir = "/content/drive/MyDrive/spaceapps-exoplanet"
os.makedirs(project_dir, exist_ok=True)

# 儲存訓練好的模型
import joblib
model_path = f"{project_dir}/best_model_v1.joblib"
joblib.dump(trained_model, model_path)
print(f"✅ 模型已存至: {model_path}")

# 下次開啟 notebook 載入模型
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("🔄 載入現有模型")
```

### 一鍵複製指令（Terminal 使用）
```bash
# 快速建立專案並推上 GitHub
git clone https://github.com/exoplanet-spaceapps/exoplanet-starter.git my-exoplanet-ai
cd my-exoplanet-ai
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# 用 Claude Code 快速開發（需要 Claude Code CLI）
claude --dangerously-skip-permissions -p "ultrathink; task: optimize BLS feature extraction for GPU batch processing; add L4 BF16 support to notebooks; commit 'feat(gpu): L4 optimization + BF16 training'"
```

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

## 🔧 **Colab 相容性問題排解**

**⚠️  重要：2025年9月 Colab 升級 NumPy 2.0，可能導致天文學套件相容性問題！**

### 快速修復
```python
# 方案 A：降版 NumPy（在第一格執行）
!pip install 'numpy==1.26.4' --force-reinstall
# ⚠️ 執行後必須重啟運行時！

# 然後安裝套件
!pip install lightkurve astroquery transitleastsquares wotan
```

### 詳細排解指南
所有相容性問題、測試腳本與解決方案請參考：
📋 **[COLAB_TROUBLESHOOTING.md](./COLAB_TROUBLESHOOTING.md)** - 完整排解指南

---

## 授權
- 程式碼：Apache License 2.0（見 LICENSE）
- 資料：依各資料源條款（NASA/MAST/Exoplanet Archive/HLSP 等）使用與引用；在論文/專案頁逐條標註來源
