# 🚀 快速開始指南 (Quick Start Guide)

## 🎯 專案目標
使用 AI 技術從 NASA 資料中尋找系外行星！

## ⚡ 30 秒快速開始

### 選項 1: Google Colab (推薦)
1. 點擊任一 notebook 連結開啟 Colab
2. 執行第一個 cell 安裝套件
3. **重要**: 手動重啟 Runtime (Runtime → Restart runtime)
4. 繼續執行剩餘 cells

### 選項 2: 本地執行
```bash
# 克隆專案
git clone https://github.com/exoplanet-spaceapps/exoplanet-starter.git
cd exoplanet-starter

# 安裝依賴 (使用 NumPy 1.26.4)
pip install -r requirements.txt

# 執行測試
python -m pytest tests/

# 開啟 Jupyter
jupyter notebook notebooks/
```

## 📚 Notebook 執行順序

### 1. **資料準備** (10 分鐘)
- `01_tap_download.ipynb` - 下載 NASA TOI 資料
- `00_verify_datasets.ipynb` - 驗證資料完整性

### 2. **基線模型** (15 分鐘)
- `02_bls_baseline.ipynb` - BLS/TLS 特徵萃取

### 3. **模型訓練** (20 分鐘)
- `03_injection_train.ipynb` - 訓練系外行星分類器
  - 選擇 Path A: 合成資料訓練
  - 或 Path B: 真實 TOI 監督學習

### 4. **推論預測** (5 分鐘)
- `04_newdata_inference.ipynb` - 對新目標進行預測

### 5. **結果分析** (5 分鐘)
- `05_metrics_dashboard.ipynb` - 查看模型表現

## ⚠️ 重要注意事項

### NumPy 2.0 相容性問題
Google Colab 預設使用 NumPy 2.0.2，但多個天文套件不相容：

**解決方案**:
1. 執行安裝 cell (會降級到 NumPy 1.26.4)
2. **手動重啟 Runtime**
3. 執行驗證 cell 確認環境正確

### GPU 加速 (可選)
如果有 GPU:
```python
# 在訓練 notebook 中
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用裝置: {device}")
```

## 🎯 範例：預測新目標

```python
from app.infer import predict_from_tic

# 預測 TIC 307210830 (已知的 TOI-5238)
result = predict_from_tic("307210830")
print(f"行星機率: {result['probability']:.2%}")
```

## 📊 預期結果

成功執行後應該看到:
- ✅ 下載 1000+ TOI 目標資料
- ✅ 萃取 14 個 BLS/TLS 特徵
- ✅ 訓練準確率 > 85%
- ✅ 產生 HTML 報告

## 🛠️ 故障排除

### 問題: ImportError with NumPy
```bash
# 解決方案
!pip install 'numpy==1.26.4' --force-reinstall
# 然後重啟 runtime
```

### 問題: 無法下載光曲線
```python
# 使用合成資料作為備案
from app.injection import generate_training_data
X, y = generate_training_data(n_samples=1000)
```

### 問題: 記憶體不足
```python
# 減少批次大小
BATCH_SIZE = 50  # 從 100 減少
```

## 📚 進階功能

### 自訂特徵
```python
from app.bls_features import extract_features
features = extract_features(time, flux, bls_result, compute_advanced=True)
```

### 機率校準
```python
from app.train import ExoplanetTrainer
trainer = ExoplanetTrainer()
trainer.calibrate_probabilities(X_val, y_val, method='isotonic')
```

### 批次處理
```python
tic_list = ["307210830", "441420236", "125819858"]
results = [predict_from_tic(tic) for tic in tic_list]
```

## 🏆 黑客松提示

1. **專注核心功能** - BLS 特徵 + 簡單分類器就很有效
2. **使用快取** - 避免重複下載相同光曲線
3. **準備展示** - 選 2-3 個確定的行星作範例
4. **記錄來源** - 引用 NASA Exoplanet Archive

## 📞 需要幫助？

- 📖 詳細文件: [README.md](README.md)
- 🐛 問題回報: [GitHub Issues](https://github.com/exoplanet-spaceapps/exoplanet-starter/issues)
- 📊 資料說明: [DATASETS.md](DATASETS.md)

---
*祝黑客松順利! Good luck with the hackathon! 🚀*