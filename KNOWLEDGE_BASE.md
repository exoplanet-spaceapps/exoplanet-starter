# 知識庫 - Exoplanet Detection 專案

## 🔧 快速解決方案索引

### NumPy 2.0 相容性問題
```python
# Google Colab 修復方案
!pip install -q numpy==1.26.4
# Runtime → Restart runtime
```

### TOI 欄位映射表
| 目標欄位 | 實際欄位 | 說明 | 單位 |
|---------|---------|------|------|
| toi_period | pl_orbper | 軌道週期 | 天 |
| toi_depth | pl_trandep | 凌日深度 | ppm |
| toi_duration | pl_trandurh | 凌日持續時間 | **小時** (需/24) |
| toi_prad | pl_rade | 行星半徑 | 地球半徑 |
| toi_insol | pl_insol | 入射流量 | 地球流量 |

### KOI False Positives 查詢
```python
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

# 獲取 KOI False Positives (替代 EB Catalog)
koi_fp = NasaExoplanetArchive.query_criteria(
    table="cumulative",
    where="koi_disposition='FALSE POSITIVE'",
    format="ipac"
)
```

### 重複欄位處理
```python
# 安全處理可能的重複欄位
if df.columns.duplicated().any():
    df = df.loc[:, ~df.columns.duplicated()]

# 使用 .values 避免 DataFrame 返回
series_data = df[column].values
```

### Google Drive 持久化
```python
from google.colab import drive
drive.mount('/content/drive')

# 儲存到 Drive
project_dir = Path('/content/drive/MyDrive/spaceapps-exoplanet')
project_dir.mkdir(parents=True, exist_ok=True)
```

### GitHub 推送 (含大檔案)
```python
# 安裝 Git LFS
!apt-get install -y git-lfs
!git lfs install
!git lfs track "*.csv"

# 推送
!git add data/*.csv data/*.json
!git commit -m "data: update NASA exoplanet data"
!git push origin main
```

## 📊 資料統計基準

### 正常資料分布
- TOI 總數: ~7,700 筆
- TOI 正樣本 (PC/CP/KP): ~5,900 筆 (77%)
- TOI 負樣本 (FP): ~1,200 筆 (15%)
- KOI False Positives: ~4,800 筆

### 資料品質標準
- period 完整性: >99%
- depth 完整性: >97%
- duration 完整性: 接近 100%

## 🌐 重要連結

### API 文件
- [NASA Exoplanet Archive API](https://exoplanetarchive.ipac.caltech.edu/docs/program_interfaces.html)
- [TOI Column Definitions](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html)
- [Astroquery Documentation](https://astroquery.readthedocs.io/)

### 資料來源
- [TOI Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)
- [KOI Cumulative Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)

## 🚨 常見錯誤與解決

### KeyError: 'tfopwg_disp'
**原因**: TOI 表格欄位名稱變更
**解決**: 使用正確的欄位名稱或檢查實際欄位

### ValueError: Cannot set DataFrame with multiple columns
**原因**: 欄位重複導致返回 DataFrame 而非 Series
**解決**: 移除重複欄位或使用 .values

### TAP Service timeout
**原因**: 查詢資料量過大
**解決**: 分批查詢或使用 format='ipac' 減少負載

### Villanova EB Catalog 無法連接
**原因**: 網站已下線 (2025)
**解決**: 使用 KOI False Positives 替代

## 💡 最佳實踐

1. **資料下載**
   - 永遠檢查欄位名稱是否存在
   - 準備備用資料來源
   - 記錄資料來源版本

2. **資料處理**
   - 明確處理 NaN 值
   - 注意單位轉換（特別是時間）
   - 保留原始欄位作為備份

3. **Colab 環境**
   - 第一步永遠處理 NumPy 版本
   - 設定資料持久化（Drive 或 GitHub）
   - 定期儲存中間結果

4. **版本控制**
   - 使用日期標記資料版本
   - 包含 Co-Authored-By 資訊
   - 記錄詳細的 commit message

## 📝 檢查清單

開始新的 Colab Session：
- [ ] 修復 NumPy 版本
- [ ] 掛載 Google Drive
- [ ] 克隆 GitHub 倉庫
- [ ] 檢查資料是否已下載
- [ ] 驗證資料完整性

資料下載後：
- [ ] 檢查樣本數量
- [ ] 驗證正負樣本比例
- [ ] 確認物理參數完整性
- [ ] 儲存到 Drive
- [ ] 建立 data_provenance.json

---
最後更新: 2025-09-30
專案: NASA Space Apps 2025 - Exoplanet Detection with AI