# 資料來源與追溯文件 (Data Provenance)

## 檔案產生時間戳
- 建立日期：2025-01-29
- 最後更新：2025-01-29
- 資料快照版本：NASA Exoplanet Archive 2025年1月版

## 主要資料來源

### 1. NASA Exoplanet Archive (TAP Service)
- **端點**: https://exoplanetarchive.ipac.caltech.edu/TAP
- **協議**: Table Access Protocol (TAP/ADQL)
- **更新頻率**:
  - TOI表：每日更新
  - pscomppars：每週更新
  - KOI/TCE：任務結束後歷史資料

#### 1.1 TOI (TESS Objects of Interest) 資料
```sql
-- 查詢時間：2025-01-29
-- 表格：toi
-- 筆數：~7000+ 候選天體
SELECT tid, toi, toipfx, tfopwg_disp, pl_orbper, pl_rade,
       pl_bmasse, st_tmag, ra, dec
FROM toi
WHERE tfopwg_disp IN ('PC', 'CP', 'KP', 'FP')
ORDER BY tid
```
- **輸出檔案**: `data/toi.csv`
- **欄位說明**: https://exoplanetarchive.ipac.caltech.edu/docs/API_toi_columns.html
- **引用**: Guerrero et al. 2021, ApJS, 254, 39

#### 1.2 Planetary Systems Composite Parameters (pscomppars)
```sql
-- 查詢時間：2025-01-29
-- 表格：pscomppars
-- 筆數：~5600+ 確認行星
SELECT pl_name, hostname, pl_orbper, pl_rade, pl_masse,
       st_teff, st_rad, disc_year, tic_id, default_flag
FROM pscomppars
WHERE disc_facility = 'Transiting Exoplanet Survey Satellite (TESS)'
  AND default_flag = 1
```
- **輸出檔案**: `data/confirmed_planets.csv`
- **欄位說明**: https://exoplanetarchive.ipac.caltech.edu/docs/API_PS_columns.html
- **引用**: Akeson et al. 2013, PASP, 125, 989

### 2. Kepler Eclipsing Binary Catalog
- **來源**: Villanova University
- **網址**: http://keplerebs.villanova.edu/
- **版本**: Version 3 (2016)
- **資料量**: 2877 個雙星系統

```python
# 使用 astroquery 獲取
from astroquery.vizier import Vizier
catalog = Vizier.get_catalogs("J/ApJ/827/50")  # Kirk et al. 2016
```
- **輸出檔案**: `data/kepler_eb.csv`
- **引用**: Kirk et al. 2016, AJ, 151, 68

### 3. MAST (Mikulski Archive for Space Telescopes)
- **端點**: https://mast.stsci.edu/api/
- **工具**: Lightkurve v2.4.0
- **資料產品**:
  - TESS SPOC v5.0 (2-minute cadence)
  - TESS HLSP QLP (Quick Look Pipeline)
  - Kepler LC/SC PDC-SAP光曲線

```python
# 光曲線下載範例
import lightkurve as lk
search_result = lk.search_lightcurve("TIC 25155310",
                                     mission="TESS",
                                     author="SPOC")
lc = search_result.download_all()
```

### 4. 合成資料生成參數
```python
# 合成注入參數範圍
injection_params = {
    "period": [0.5, 50.0],      # 天
    "depth": [0.0001, 0.05],    # 凌日深度
    "duration": [0.5, 24.0],     # 小時
    "snr": [5, 100],             # 訊噪比
}
```
- **輸出檔案**: `data/synthetic_training.csv`
- **資料量**: 10,000 個合成樣本（5000正/5000負）

## 資料處理與版本控制

### 預處理步驟
1. **去趨勢**: Wotan `flatten(window_length=401)`
2. **異常值移除**: 3-sigma clipping
3. **歸一化**: 中位數歸一化至 1.0
4. **品質過濾**:
   - 移除 NaN 值
   - 要求最少 100 個有效資料點
   - SNR > 5 for training

### 版本追蹤
```json
{
  "data_version": "2025.01.29",
  "toi_last_update": "2025-01-28T12:00:00Z",
  "kepler_eb_version": "V3_2016",
  "lightkurve_version": "2.4.0",
  "astroquery_version": "0.4.6",
  "processing_date": "2025-01-29"
}
```

## 資料使用限制與偏差

### 已知限制
1. **週期偏差**: 最佳化於 0.5-20 天週期
2. **深度敏感度**: 深度 < 500 ppm 偵測率降低
3. **恆星類型**: 主要訓練於 FGK 型恆星
4. **觀測時長**: 需要至少 2 個凌日事件

### 潛在偏差來源
1. **TESS 觀測策略**: 南北半球覆蓋不均
2. **Kepler 視場**: 僅天鵝座區域
3. **亮度偏差**: TESS tmag < 16 的目標
4. **確認偏差**: 大型行星更容易被確認

## 資料引用要求

使用本資料集請引用：

### 主要資料來源
```bibtex
@article{exoplanetarchive2013,
  author = {Akeson, R. L. and others},
  title = {The NASA Exoplanet Archive},
  journal = {PASP},
  volume = {125},
  pages = {989},
  year = {2013},
  doi = {10.1086/672273}
}

@article{tess2015,
  author = {Ricker, G. R. and others},
  title = {TESS Mission},
  journal = {JATIS},
  volume = {1},
  pages = {014003},
  year = {2015},
  doi = {10.1117/1.JATIS.1.1.014003}
}

@article{keplerEB2016,
  author = {Kirk, B. and others},
  title = {Kepler Eclipsing Binary Catalog},
  journal = {AJ},
  volume = {151},
  pages = {68},
  year = {2016},
  doi = {10.3847/0004-6256/151/3/68}
}
```

### 軟體工具
```bibtex
@software{lightkurve2018,
  author = {Lightkurve Collaboration},
  title = {Lightkurve: Kepler and TESS time series analysis},
  year = {2018},
  url = {https://github.com/lightkurve/lightkurve}
}

@software{astroquery2019,
  author = {Ginsburg, A. and others},
  title = {astroquery: Astronomical database queries},
  year = {2019},
  doi = {10.3847/1538-3881/aafc33}
}
```

## 資料存取日誌

| 日期 | 資料源 | 查詢類型 | 筆數 | 檔案 |
|------|---------|----------|------|------|
| 2025-01-29 | NASA TAP | TOI候選 | 7234 | toi.csv |
| 2025-01-29 | NASA TAP | 確認行星 | 5657 | confirmed_planets.csv |
| 2025-01-29 | Villanova | Kepler EB | 2877 | kepler_eb.csv |
| 2025-01-29 | Synthetic | 訓練資料 | 10000 | synthetic_training.csv |

## 資料完整性檢查

```python
# 驗證碼
import hashlib

def verify_data_integrity():
    checksums = {
        "toi.csv": "SHA256_HASH_HERE",
        "confirmed_planets.csv": "SHA256_HASH_HERE",
        "kepler_eb.csv": "SHA256_HASH_HERE"
    }

    for file, expected_hash in checksums.items():
        with open(f"data/{file}", "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
            assert actual_hash == expected_hash, f"{file} 完整性檢查失敗"
```

## 聯絡與支援

- **NASA Exoplanet Archive**: exoplanetarchive@ipac.caltech.edu
- **MAST Archive**: archive@stsci.edu
- **專案維護者**: hctsai1006 <39769660@cuni.cz>

---

*本文件最後更新：2025-01-29*
*下次計畫更新：2025-02-15 (TESS Sector 71 資料釋出後)*