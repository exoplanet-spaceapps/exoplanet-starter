# 開發日誌 - NASA Space Apps 2025 Exoplanet Detection

## 2025-09-30 Session Summary

### 🎯 主要成就
成功解決 TOI 資料下載的所有技術問題，建立完整的資料管線。

### 🔧 技術問題與解決方案

#### 1. TOI 欄位名稱映射問題
**問題**: TOI 表格使用 `pl_*` 前綴而非預期的 `toi_*`
```python
# 錯誤的欄位名稱
'toi_period', 'toi_depth', 'toi_duration'

# 正確的欄位名稱
'pl_orbper', 'pl_trandep', 'pl_trandurh'
```

**解決方案**: 建立欄位映射字典
```python
column_mapping = {
    'toi_period': 'pl_orbper',      # 軌道週期 (天)
    'toi_depth': 'pl_trandep',       # 凌日深度 (ppm)
    'toi_duration': 'pl_trandurh',   # 凌日持續時間 (小時→需/24轉天)
}
```

#### 2. Villanova EB Catalog 無法存取
**問題**: Kepler Eclipsing Binary Catalog (http://keplerebs.villanova.edu/) 無法連接

**解決方案**: 使用 KOI False Positives 作為替代
```python
# 從 NASA Archive 獲取 KOI False Positives
koi_fp = NasaExoplanetArchive.query_criteria(
    table="cumulative",
    where="koi_disposition='FALSE POSITIVE'",
    format="ipac"
)
# 成功獲取 4,839 筆負樣本資料
```

#### 3. NumPy 2.0 相容性問題
**問題**: Google Colab 使用 NumPy 2.0.2，但天文套件不相容

**解決方案**:
```bash
!pip install -q numpy==1.26.4
# 然後 Runtime → Restart runtime
```

#### 4. 重複欄位錯誤
**問題**: `ValueError: Cannot set a DataFrame with multiple columns to the single column period`

**解決方案**:
```python
# 檢查並移除重複欄位
if eb_df_processed.columns.duplicated().any():
    eb_df_processed = eb_df_processed.loc[:, ~eb_df_processed.columns.duplicated()]

# 使用 .values 避免 DataFrame 返回
negative_samples_koi[param] = col_data.values
```

#### 5. 資料持久化問題
**問題**: Colab 資料只存在暫時目錄，重啟後消失

**解決方案**: 實作三種持久化方式
1. Google Drive 自動掛載與版本化儲存
2. GitHub 推送（含 Git LFS 支援）
3. 資料完整性驗證

### 📊 資料統計

**最終資料集**:
- 總樣本數: 11,979 筆
- 正樣本 (行星候選): 5,944 筆 (49.6%)
- 負樣本 (False Positives): 6,035 筆 (50.4%)
  - TOI FP: 1,196 筆
  - KOI FP: 4,839 筆

**資料品質**:
- period: 99.2% 完整
- depth: 97.9% 完整
- duration: 100% 完整

### 💡 關鍵學習

1. **NASA Archive API 細節**
   - TOI 表格欄位命名規則：使用 `pl_*` 前綴代表 planetary parameters
   - 單位轉換：`pl_trandurh` 是小時單位，需要 /24 轉換為天

2. **資料來源備案策略**
   - 主要來源失敗時，需要預先準備替代方案
   - KOI False Positives 是優質的 EB 負樣本來源

3. **Colab 環境特性**
   - 必須考慮資料持久化
   - NumPy 版本衝突需要特別處理
   - Google Drive 整合是最方便的持久化方案

### 🔗 參考資源

- [NASA Exoplanet Archive TOI Columns](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html)
- [TOI_DATA_SOLUTION.md](./TOI_DATA_SOLUTION.md) - 問題診斷完整文件
- [Kirk et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016AJ....151...68K) - Kepler EB Catalog

### 📝 待辦事項

- [x] 修復 TOI 欄位映射問題
- [x] 找到 EB Catalog 替代方案
- [x] 處理重複欄位錯誤
- [x] 實作資料持久化
- [ ] 執行 02_bls_baseline.ipynb
- [ ] 從光曲線計算缺失的物理參數
- [ ] 實作 SMOTE 資料平衡

### 🏆 成果

1. **01_tap_download.ipynb** - 完整修復並增強
   - 正確的欄位映射
   - 多重資料來源備案
   - 完整的持久化方案

2. **TOI_DATA_SOLUTION.md** - 問題診斷文件
   - 詳細記錄所有問題與解決方案
   - 可作為未來參考

3. **資料集準備完成**
   - 11,979 筆真實 NASA 觀測資料
   - 無任何模擬資料
   - 正負樣本平衡良好

---

## Co-Authors
- hctsai1006 <39769660@cuni.cz>
- Claude (Anthropic)

## 專案倉庫
https://github.com/exoplanet-spaceapps/exoplanet-starter