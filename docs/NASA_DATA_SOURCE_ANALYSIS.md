# NASA Space Apps Challenge 2025 - 系外行星挑戰資料來源分析報告

## 📋 挑戰基本資訊

**挑戰名稱**: A World Away: Hunting for Exoplanets with AI
**挑戰連結**: https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/
**分析日期**: 2025年10月4日
**報告版本**: 1.0

---

## 🎯 挑戰目標

創建一個基於 NASA 開源系外行星數據集的 AI/機器學習模型，能夠：
- 分析太空望遠鏡觀測數據
- 識別潛在的系外行星
- 提供網頁介面供使用者互動
- （可選）允許上傳新數據進行分析
- （可選）支援模型持續訓練與優化

---

## 🗂️ NASA 官方數據資源完整清單

### 1️⃣ 主要數據集

#### 🔭 Kepler Objects of Interest (KOI) 數據集

**官方連結**:
```
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
```

**數據集特色**:
- 資料來源: NASA Kepler 太空望遠鏡
- 觀測期間: 2009-2018年
- 候選行星數量: 數千個
- 資料格式: CSV、JSON、VOTable
- 資料內容: 光變曲線、行星參數、恆星參數

**適用場景**:
- 訓練系外行星偵測模型
- 凌日訊號特徵學習
- 假陽性樣本分析

---

#### 🛰️ TESS Objects of Interest (TOI) 數據集

**官方連結**:
```
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI
```

**數據集特色**:
- 資料來源: NASA TESS (Transiting Exoplanet Survey Satellite)
- 觀測狀態: 持續進行中
- 候選行星數量: 7,703+ 個（截至 2025年9月28日）
- 更新頻率: 定期更新
- 資料格式: CSV、JSON、VOTable

**適用場景**:
- 最新系外行星候選數據
- 模型驗證與測試
- 即時數據分析應用

---

#### 🔬 K2 Planets and Candidates 數據集

**官方連結**:
```
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc
```

**數據集特色**:
- 資料來源: Kepler K2 延伸任務
- 觀測範圍: 多個天區的觀測數據
- 資料特色: 涵蓋不同類型的恆星系統
- 資料格式: CSV、JSON、VOTable

**適用場景**:
- 擴充訓練數據多樣性
- 不同恆星類型的行星偵測

---

### 2️⃣ 已確認行星數據

**官方確認行星數據庫**:
```
https://exoplanetarchive.ipac.caltech.edu/
```

**統計數據**:
- 已確認系外行星: 6,022 顆（截至 2025年10月2日）
- 資料完整性: 包含軌道參數、物理性質、發現方法等

---

### 3️⃣ NASA Exoplanet Archive 主入口

#### 🏠 官方首頁
```
https://exoplanetarchive.ipac.caltech.edu/
```

**提供服務**:
- 互動式資料表搜尋
- 批量資料下載
- API 程式化存取
- 資料視覺化工具
- 線上分析工具

---

#### 📦 批量資料下載
```
https://exoplanetarchive.ipac.caltech.edu/bulk_data_download
```

**下載選項**:
- 完整數據集打包下載
- 支援格式: CSV、JSON、FITS、VOTable
- 包含所有任務的觀測數據
- 定期更新的資料版本

---

### 4️⃣ API 與程式化存取介面

#### 🔌 TAP 服務 (Table Access Protocol)

**文檔連結**:
```
https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
```

**功能特色**:
- 使用 ADQL (Astronomical Data Query Language) 查詢
- 支援複雜的 SQL 查詢語法
- 可跨資料表聯合查詢
- 輸出格式: VOTable、CSV、JSON、TSV

**應用範例**:
```python
# Python 範例
from astroquery.ipac.nexsci import Exoplanet

# 查詢 TESS 候選行星
toi_data = Exoplanet.query_criteria(
    table='toi',
    select='*',
    where='toi_disposition="PC"'
)
```

---

#### 🖥️ API 程式介面

**文檔連結**:
```
https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html
```

**支援語言**:
- Python (推薦使用 `astroquery` 套件)
- R
- IDL
- 任何支援 HTTP 請求的語言

**API 端點範例**:
```
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI
```

**查詢參數**:
- `table`: 資料表名稱 (koi, toi, k2pandc 等)
- `select`: 欲取得的欄位
- `where`: 篩選條件
- `format`: 輸出格式 (json, csv, xml 等)

---

### 5️⃣ 資料分析工具

#### 🛠️ Transit and Ephemeris Service (凌日與星曆服務)

**工具連結**:
```
https://exoplanetarchive.ipac.caltech.edu/applications/DocSet/index.html
```

**功能**:
- 計算行星凌日時間
- 預測觀測視窗
- 星曆資料查詢

---

#### 📊 Periodogram (週期圖分析工具)

**功能**:
- 光變曲線週期性分析
- 頻率功率譜計算
- 訊號週期偵測

---

#### 🎯 EXOFAST (凌日與徑向速度擬合工具)

**功能**:
- 專業行星參數擬合
- 貝氏統計分析
- 多種觀測數據聯合分析

---

### 6️⃣ 輔助資源與平台

#### 🌐 ExoFOP (Follow-up Observing Program)

**平台連結**:
```
https://exofop.ipac.caltech.edu/
```

**功能**:
- 系外行星候選觀測協作平台
- 後續觀測資料共享
- 候選狀態追蹤

---

#### 🚀 NASA Exoplanet Exploration

**官方網站**:
```
https://exoplanets.nasa.gov/
```

**內容**:
- 系外行星探索科普資訊
- 互動式視覺化工具
- 教育資源與教材
- 最新發現與新聞

---

### 7️⃣ 學術研究論文資源

#### 📚 推薦研究文獻

1. **"Exoplanet Detection Using Machine Learning"**
   - 主題: 機器學習在系外行星偵測的應用
   - 相關技術: CNN、RNN、特徵工程

2. **"Assessment of Ensemble-Based Machine Learning Algorithms for Exoplanet Identification"**
   - 主題: 集成學習演算法評估
   - 相關技術: Random Forest、XGBoost、Stacking

---

### 8️⃣ 國際合作夥伴資源

#### 🇨🇦 Canadian Space Agency (CSA)
- NEOSSat 天文觀測數據

#### 🔭 James Webb Space Telescope (JWST)
- 系外行星大氣光譜數據
- 高解析度觀測資料

---

## 📝 資料來源引用格式建議

### 標準引用格式

```
1. Kepler Objects of Interest (KOI)
   來源: NASA Exoplanet Archive
   網址: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
   存取日期: 2025年10月4日

2. TESS Objects of Interest (TOI)
   來源: NASA Exoplanet Archive
   網址: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI
   存取日期: 2025年10月4日

3. K2 Planets and Candidates
   來源: NASA Exoplanet Archive
   網址: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc
   存取日期: 2025年10月4日

4. NASA Exoplanet Archive API
   來源: NASA Exoplanet Science Institute
   網址: https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html
   存取日期: 2025年10月4日
```

---

### 學術論文引用格式 (APA)

```
NASA Exoplanet Science Institute. (2025). NASA Exoplanet Archive.
Retrieved from https://exoplanetarchive.ipac.caltech.edu/
```

---

## 🎯 專案開發建議

### 必要功能清單

✅ **核心功能**:
1. AI/ML 模型開發
   - 使用 Kepler、TESS、K2 數據集訓練
   - 實作深度學習架構 (CNN/RNN/Transformer)
   - 達到可接受的準確率（建議 >90%）

2. Web 使用者介面
   - 資料上傳功能
   - 即時預測結果顯示
   - 視覺化光變曲線

3. 資料處理管線
   - 自動化資料前處理
   - 特徵工程
   - 資料增強技術

---

### 進階功能選項

🔄 **加分項目**:
1. 使用者自訂資料分析
   - 允許上傳 FITS/CSV 格式光變曲線
   - 即時預測與視覺化

2. 模型持續訓練
   - 線上學習機制
   - 使用者回饋整合

3. 模型效能展示
   - 準確率、召回率、F1-score
   - ROC 曲線與混淆矩陣
   - 超參數調整介面

4. 多模型比較
   - 不同演算法效能對比
   - 集成學習實作

---

## 📊 資料統計摘要

| 資料集 | 來源 | 候選/確認數量 | 更新狀態 |
|--------|------|---------------|----------|
| KOI | Kepler | 數千個候選 | 已完成任務 |
| TOI | TESS | 7,703+ 候選 | 持續更新 |
| K2 | Kepler K2 | 數百個候選 | 已完成任務 |
| 確認行星 | 綜合 | 6,022 顆 | 持續更新 |

---

## 🔗 快速存取連結整理

### 資料下載
- KOI: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
- TOI: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI
- K2: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc

### 文件與工具
- 主入口: https://exoplanetarchive.ipac.caltech.edu/
- API 文件: https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html
- TAP 服務: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html
- 批量下載: https://exoplanetarchive.ipac.caltech.edu/bulk_data_download

### 教育資源
- NASA Exoplanets: https://exoplanets.nasa.gov/
- ExoFOP: https://exofop.ipac.caltech.edu/

---

## 📌 重要備註

1. **資料使用授權**: NASA 資料為公開資料，可自由使用於教育與研究目的
2. **引用要求**: 建議在專案中明確標示資料來源
3. **資料更新**: TESS 資料持續更新，建議定期檢查最新版本
4. **技術支援**: NASA Exoplanet Archive 提供技術支援與諮詢服務

---

## 📧 聯絡資訊

**NASA Exoplanet Archive 技術支援**:
- 網站: https://exoplanetarchive.ipac.caltech.edu/
- 透過官網提供的聯絡表單提交問題

---

**文件結束**

---

**附註**: 本報告由 Claude Code 協助分析與整理，所有連結與資訊均來自 NASA 官方資源。建議在使用前再次確認連結有效性。
