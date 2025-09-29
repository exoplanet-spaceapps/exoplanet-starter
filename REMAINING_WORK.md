# 🚧 剩餘工作清單 (Remaining Work)

## 📊 專案完成度總覽

| 階段 | 完成度 | 狀態 |
|------|--------|------|
| **Phase 1** - BLS/TLS 基線 | 95% | ✅ 核心完成 |
| **Phase 2** - 合成注入 | 90% | ✅ 核心完成 |
| **Phase 3** - 監督學習 | 85% | ✅ 核心完成 |
| **Phase 4** - 推論管線 | 90% | ✅ 核心完成 |
| **Phase 5** - 評估文件 | 95% | ✅ 完成 |
| **Phase 6** - Colab 整合 | 80% | ⚠️ 需實測 |
| **整體測試覆蓋** | 40% | ❌ 待補充 |

## 🔧 立即需要修復的問題

### 1. **01_tap_download.ipynb 依賴錯誤** 🚨
```python
NameError: name 'astroquery' is not defined
```
**狀態**: ✅ 已修復 - 加入 NumPy<2.0 限制與正確的 import 檢查

### 2. **NumPy 2.0 相容性問題** ⚠️
- Google Colab 現使用 NumPy 2.0.2
- 多個天文學套件不相容
- **解決方案**: 在所有 notebooks 第一格加入 `!pip install 'numpy<2.0'`

## 📝 尚未完成的核心工作

### 🧪 **單元測試** (Priority: HIGH)
- [x] `test_injection.py` - 185行，已完成
- [x] `test_bls_features.py` - 剛建立，完整測試
- [ ] `test_train.py` - **待建立**
- [ ] `test_infer.py` - **待建立**
- [ ] `test_report.py` - **待建立**
- [ ] `test_utils.py` - **待建立**

### 🔬 **端對端測試** (Priority: CRITICAL)
- [ ] 在實際 Google Colab 環境執行所有 notebooks
- [ ] 驗證依賴套件安裝流程
- [ ] 測試 GPU 加速（如果可用）
- [ ] 驗證資料下載與處理
- [ ] 確認模型訓練與儲存
- [ ] 測試推論管線完整流程

### 📚 **文件與範例** (Priority: MEDIUM)
- [ ] 建立快速開始指南 (Quick Start Guide)
- [ ] 添加更多程式碼註解
- [ ] 建立 API 文件
- [ ] 添加故障排除章節
- [ ] 建立效能基準測試報告

## 🎯 建議執行順序

### **Step 1: 修復依賴問題** (30分鐘)
```bash
# 在所有 notebooks 開頭加入
!pip install 'numpy==1.26.4' --force-reinstall
!pip install lightkurve astroquery transitleastsquares wotan
```

### **Step 2: 完成關鍵測試** (2小時)
```python
# 建立 test_train.py
# 測試 ExoplanetTrainer 類別
# 測試模型儲存與載入
# 測試超參數調優
```

### **Step 3: Colab 實測** (1小時)
1. 開啟 Google Colab
2. 依序執行每個 notebook
3. 記錄任何錯誤或警告
4. 調整依賴安裝順序

### **Step 4: 整合測試腳本** (1小時)
```python
# integration_test.py
def test_full_pipeline():
    # 1. 下載資料
    # 2. 產生合成資料
    # 3. 訓練模型
    # 4. 執行推論
    # 5. 產生報告
    pass
```

## 🐛 已知問題與限制

### **依賴相容性**
| 套件 | NumPy 2.0 相容性 | 解決方案 |
|------|------------------|----------|
| lightkurve | ✅ 支援 | 最新版本 OK |
| astroquery | ⚠️ 部分 | 可能需要特定版本 |
| transitleastsquares | ❌ 不支援 | 必須用 NumPy<2.0 |
| wotan | ❌ 未知 | 建議用 NumPy<2.0 |

### **功能限制**
- 批次處理可能遇到記憶體限制（>1000 光曲線）
- BLS 搜尋對長週期（>20天）效率較低
- 合成注入可能無法完全模擬儀器效應

### **效能考量**
- 單一光曲線處理時間：~5-10秒
- 批次 100 個目標：~10-15分鐘
- 訓練 10000 樣本：~5分鐘（CPU）

## ✅ 完成檢查清單

在提交專案前，請確認：

- [ ] 所有 notebooks 可在 Colab 執行
- [ ] 依賴安裝指令正確無誤
- [ ] 核心功能都有基本測試
- [ ] README 包含所有必要資訊
- [ ] 資料來源都有適當引用
- [ ] 授權資訊正確（Apache-2.0）
- [ ] 移除所有除錯輸出
- [ ] 確認沒有硬編碼路徑
- [ ] 檢查沒有敏感資訊洩露

## 🚀 快速修復指令

### 修復所有 notebooks 的依賴
```python
# 貼到每個 notebook 第一格
import subprocess
import sys

def safe_install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

# 核心套件（固定版本）
safe_install("numpy==1.26.4")
safe_install("scipy==1.11.4")
safe_install("matplotlib>=3.7.0")
safe_install("pandas>=2.0.0")
safe_install("scikit-learn>=1.3.0")

# 天文學套件
safe_install("astropy>=5.3.0")
safe_install("astroquery>=0.4.6")
safe_install("lightkurve>=2.4.0")

# 進階套件
safe_install("transitleastsquares")
safe_install("wotan")

print("✅ 所有套件安裝完成")
```

## 📅 時間估算

| 任務 | 預估時間 | 優先級 |
|------|----------|--------|
| 修復依賴問題 | 30分鐘 | 🔴 緊急 |
| 完成單元測試 | 2-3小時 | 🟠 高 |
| Colab 實測 | 1-2小時 | 🔴 緊急 |
| 整合測試 | 1小時 | 🟠 高 |
| 文件完善 | 1-2小時 | 🟡 中 |
| **總計** | **6-8小時** | - |

## 💡 建議

1. **優先處理 Colab 相容性** - 這是使用者第一個接觸點
2. **專注核心功能測試** - 不需要 100% 覆蓋率
3. **保持簡單** - 黑客松專案不需要過度工程化
4. **準備演示** - 確保有 1-2 個完美運作的範例

---

*最後更新：2025-09-29*
*專案狀態：可運作但需要測試驗證*