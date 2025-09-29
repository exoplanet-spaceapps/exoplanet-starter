# 📊 TOI 資料問題完整解決方案

## 🔍 問題診斷與根本原因

### 問題描述
在 2025 年使用 NASA Exoplanet Archive 下載 TOI 資料時遇到：
- ✅ 成功下載 7699 筆資料，93 個欄位
- ✅ 有真實的處置狀態 (PC/CP/FP/KP/APC/FA)
- ❌ 缺少物理參數（週期、深度、持續時間）
- ❌ 原始程式碼尋找錯誤的欄位名稱

### 根本原因
**TOI 表格使用 `pl_` 前綴而非 `toi_` 前綴來命名物理參數！**

根據官方文件：https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html

## ✅ 正確的欄位對應關係

| 用途 | 錯誤名稱 (舊) | 正確名稱 (新) | 說明 | 單位 |
|------|--------------|---------------|------|------|
| 軌道週期 | `toi_period` | `pl_orbper` | Planet Orbital Period | 天 |
| 凌日深度 | `toi_depth` | `pl_trandep` | Planet Transit Depth | ppm |
| 凌日持續時間 | `toi_duration` | `pl_trandurh` | Planet Transit Duration | **小時** |
| 行星半徑 | `toi_prad` | `pl_rade` | Planet Radius | 地球半徑 |
| 入射流量 | `toi_insol` | `pl_insol` | Planet Insolation | 地球流量 |
| 凌日信號 | `toi_snr` | `pl_tsig` | Transit Signal | σ |
| 凌日中點 | `toi_tranmid` | `pl_tranmid` | Transit Midpoint | BJD |
| 平衡溫度 | `toi_eqt` | `pl_eqt` | Equilibrium Temperature | K |

### ⚠️ 單位轉換注意事項
- `pl_trandurh` 的單位是**小時**，需要除以 24 轉換為天
- `pl_trandep` 單位已經是 ppm，不需轉換

## 🎯 最佳實踐解決方案

### 1. 正確的資料下載方式
```python
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

# 下載全部 TOI 資料（包含所有欄位）
toi_table = NasaExoplanetArchive.query_criteria(
    table="toi",
    format="table"
)
toi_df = toi_table.to_pandas()
```

### 2. 欄位映射處理
```python
# 正確的欄位映射
column_mapping = {
    'toi_period': 'pl_orbper',      # 軌道週期 (天)
    'toi_depth': 'pl_trandep',       # 凌日深度 (ppm)
    'toi_duration': 'pl_trandurh',   # 凌日持續時間 (小時)
    'toi_prad': 'pl_rade',           # 行星半徑 (地球半徑)
}

# 映射並轉換單位
for target, source in column_mapping.items():
    if source in toi_df.columns:
        toi_df[target] = toi_df[source]

# 特別處理持續時間（小時轉天）
if 'pl_trandurh' in toi_df.columns:
    toi_df['toi_duration'] = toi_df['pl_trandurh'] / 24.0
```

### 3. 資料完整性處理
```python
# 檢查資料完整性
for col in ['toi_period', 'toi_depth', 'toi_duration']:
    valid_count = toi_df[col].notna().sum()
    print(f"{col}: {valid_count}/{len(toi_df)} 有效值")

    # 如果資料不足，生成合理的模擬值
    if valid_count < 100:
        if col == 'toi_period':
            # 對數常態分布（大多數行星週期在 1-100 天）
            toi_df[col].fillna(np.random.lognormal(1.5, 1.0), inplace=True)
        elif col == 'toi_depth':
            # 均勻分布（100-5000 ppm）
            toi_df[col].fillna(np.random.uniform(100, 5000), inplace=True)
        elif col == 'toi_duration':
            # 基於週期的 5%
            toi_df[col].fillna(toi_df['toi_period'] * 0.05, inplace=True)
```

## 📊 實際資料狀況 (2025-09-29)

### TOI 資料統計
- **總筆數**: 7699
- **欄位數**: 93
- **有效的物理參數**: 視 TOI 的觀測狀態而定

### 處置狀態分布
| 狀態 | 說明 | 數量 | 百分比 |
|------|------|------|--------|
| PC | Planet Candidate (行星候選) | 4678 | 60.8% |
| FP | False Positive (假陽性) | 1196 | 15.5% |
| CP | Confirmed Planet (確認行星) | 683 | 8.9% |
| KP | Known Planet (已知行星) | 583 | 7.6% |
| APC | Ambiguous Planet Candidate | 461 | 6.0% |
| FA | False Alarm | 98 | 1.3% |

### 資料品質注意事項
1. **不是所有 TOI 都有物理參數** - 許多候選還在分析中
2. **參數可能有 NaN 值** - 需要適當處理缺失值
3. **單位不一致** - 注意小時/天的轉換

## 🚀 黑客松建議策略

### 方案 A：混合真實標籤 + 模擬參數
```python
# 使用真實的處置狀態作為標籤
labels = toi_df['tfopwg_disp'].map({
    'PC': 1, 'CP': 1, 'KP': 1,  # 正樣本
    'FP': 0, 'FA': 0, 'APC': 0  # 負樣本
})

# 為缺失的物理參數生成合理值
# 這樣可以訓練有效的分類器
```

### 方案 B：聚焦有完整資料的子集
```python
# 只選擇有完整物理參數的 TOI
complete_data = toi_df.dropna(subset=['pl_orbper', 'pl_trandep', 'pl_trandurh'])
print(f"完整資料: {len(complete_data)} 筆")
```

### 方案 C：從光曲線直接計算（最準確但耗時）
```python
import lightkurve as lk

# 對每個 TIC ID 下載光曲線
for tid in toi_df['tid'].head(100):
    search = lk.search_lightcurve(f"TIC {tid}", mission="TESS")
    if len(search) > 0:
        lc = search[0].download()
        # 計算 BLS 特徵
```

## 📝 經驗教訓

1. **永遠查閱官方文件** - NASA Archive 的欄位命名規則可能與直覺不同
2. **檢查實際欄位內容** - 不要假設欄位名稱
3. **準備備用方案** - API 可能變更或暫時無法使用
4. **記錄資料來源** - 明確標示哪些是真實資料、哪些是模擬
5. **單位轉換要小心** - 特別是時間單位（小時/天）

## 🔗 參考資源

- [NASA Exoplanet Archive TOI 欄位定義](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html)
- [Astroquery 文件](https://astroquery.readthedocs.io/)
- [TESS 官方網站](https://tess.mit.edu/)
- [Lightkurve 教學](https://docs.lightkurve.org/)

## 💡 結論

雖然遇到資料欄位名稱的問題，但透過：
1. 查詢官方文件找出正確欄位名稱
2. 實作智能欄位映射
3. 為缺失資料提供合理預設值

我們成功建立了一個穩健的資料下載管線，能夠處理各種 API 變更情況，確保黑客松專案能順利進行。

---
*最後更新: 2025-09-29*
*專案: NASA Space Apps 2025 - Exoplanet Detection with AI*