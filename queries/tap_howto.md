# TAP 使用小抄

## NASA Exoplanet Archive TAP 服務指南

### 端點 URLs
- **同步查詢**: https://exoplanetarchive.ipac.caltech.edu/TAP/sync
- **非同步查詢**: https://exoplanetarchive.ipac.caltech.edu/TAP/async
- **文檔**: https://exoplanetarchive.ipac.caltech.edu/docs/TAP/usingTAP.html

### Python 中使用 TAP（同步查詢）
```python
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

# 方法 1: 使用 query_criteria（簡單查詢）
toi_table = NasaExoplanetArchive.query_criteria(
    table="toi",
    where="tfopwg_disp IN ('PC', 'CP', 'KP')",
    select="tid,toi,pl_orbper,pl_rade"
)

# 方法 2: 使用原始 SQL（複雜查詢）
query = """
SELECT tid, toi, pl_orbper, pl_rade, st_tmag
FROM toi
WHERE tfopwg_disp = 'PC' AND pl_rade < 2.0
ORDER BY pl_rade
"""
result = NasaExoplanetArchive.query_sql(query)

# 方法 3: 使用 pyvo TAP 客戶端
import pyvo as vo
tap_service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
result = tap_service.search(query)
```

### 使用 curl 進行查詢
```bash
# 基本 TAP 查詢
curl -X POST "https://exoplanetarchive.ipac.caltech.edu/TAP/sync" \
     -d "REQUEST=doQuery" \
     -d "LANG=ADQL" \
     -d "FORMAT=csv" \
     -d "QUERY=SELECT+*+FROM+toi+WHERE+tfopwg_disp='PC'"

# 下載為檔案
curl -X POST "https://exoplanetarchive.ipac.caltech.edu/TAP/sync" \
     -d "REQUEST=doQuery" \
     -d "LANG=ADQL" \
     -d "FORMAT=csv" \
     -d "QUERY=SELECT+*+FROM+toi+WHERE+tfopwg_disp='PC'" \
     -o toi_candidates.csv
```

### 非同步查詢（大資料集）
```python
import requests
import time

# 1. 提交非同步任務
async_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/async"
query = "SELECT * FROM pscomppars WHERE discoverymethod='Transit'"
response = requests.post(async_url, data={
    'REQUEST': 'doQuery',
    'LANG': 'ADQL',
    'FORMAT': 'csv',
    'QUERY': query
})
job_url = response.headers['Location']

# 2. 開始執行
requests.post(f"{job_url}/phase", data={'PHASE': 'RUN'})

# 3. 檢查狀態
while True:
    status = requests.get(f"{job_url}/phase").text
    if status == 'COMPLETED':
        break
    time.sleep(5)

# 4. 獲取結果
result_url = f"{job_url}/results/result"
data = requests.get(result_url).text
```

### 格式選項
- `FORMAT=csv` - CSV 格式（推薦用於資料分析）
- `FORMAT=json` - JSON 格式
- `FORMAT=votable` - VOTable XML 格式
- `FORMAT=tsv` - Tab 分隔格式
- `FORMAT=ascii` - ASCII 表格

### 常用篩選條件
```sql
-- 數值範圍
WHERE pl_rade BETWEEN 0.8 AND 1.5

-- 多值匹配
WHERE tfopwg_disp IN ('PC', 'CP', 'KP')

-- 非空值
WHERE pl_orbper IS NOT NULL

-- 字串匹配
WHERE hostname LIKE 'Kepler-%'

-- 組合條件
WHERE pl_rade < 2.0 AND pl_orbper < 10 AND tfopwg_disp = 'PC'
```

### 效能優化建議
1. **使用 SELECT 指定欄位**而非 SELECT *
2. **加入 LIMIT** 限制回傳筆數
3. **使用索引欄位**進行篩選（如 tid, toi, pl_name）
4. **批次查詢**大資料集時使用非同步
5. **本地快取**常用查詢結果

### 錯誤處理
```python
try:
    result = NasaExoplanetArchive.query_criteria(
        table="toi",
        where="tfopwg_disp='PC'"
    )
except Exception as e:
    print(f"查詢失敗: {e}")
    # 使用備份資料或重試
```

### 資料更新頻率
- TOI 表：每日更新
- pscomppars：每週更新
- 其他表：依任務而定

### 注意事項
- 同步查詢限制：50,000 筆
- 超時限制：600 秒
- 並發限制：建議間隔 1 秒
- 使用 `default_flag=1` 獲取首選參數集