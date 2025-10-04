# NASA 資料來源快速參考指南

## 🚀 三大核心數據集（必備）

### 1. Kepler (KOI)
```
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
```
- 數千個候選系外行星
- 2009-2018 年觀測數據
- 適合: 模型訓練基礎數據

---

### 2. TESS (TOI) ⭐ 推薦
```
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI
```
- **7,703+ 候選行星**（持續更新）
- 最新觀測數據
- 適合: 模型驗證與測試

---

### 3. K2
```
https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc
```
- 數百個候選行星
- 多樣性天區觀測
- 適合: 擴充訓練數據

---

## 🔌 程式化存取（推薦用於專案）

### API 文件
```
https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html
```

### Python 範例
```python
from astroquery.ipac.nexsci import Exoplanet

# 取得 TOI 數據
data = Exoplanet.query_criteria(
    table='toi',
    select='*',
    where='toi_disposition="PC"'
)
```

---

## 📦 批量下載
```
https://exoplanetarchive.ipac.caltech.edu/bulk_data_download
```
- 所有數據集完整打包
- 支援 CSV、JSON、FITS

---

## 🏠 主要入口
```
https://exoplanetarchive.ipac.caltech.edu/
```
- 已確認行星: **6,022 顆**
- 互動式搜尋工具
- 線上分析功能

---

## 📝 標準引用格式（提交專案必備）

```
資料來源:
1. NASA Exoplanet Archive - Kepler Objects of Interest
   https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative

2. NASA Exoplanet Archive - TESS Objects of Interest
   https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI

3. NASA Exoplanet Archive - K2 Planets and Candidates
   https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc

存取日期: 2025年10月4日
```

---

## 💡 快速開始建議

1. **先下載 TOI 數據集**（最新、數量最多）
2. **使用 API 進行程式化存取**（自動化處理）
3. **結合 Kepler 數據**（增加訓練樣本）
4. **記錄所有資料來源**（競賽要求）

---

## ⚠️ 重要提醒

- ✅ 所有 NASA 資料可自由使用
- ✅ 需明確標示資料來源
- ✅ TESS 數據持續更新
- ✅ 建議使用最新版本數據

---

詳細資訊請參考: `NASA資料來源分析報告.md`
