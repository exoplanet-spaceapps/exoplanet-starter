# 🔍 GitHub 推送問題完整診斷報告

## 問題描述
Commit 353a881 推送成功，但 GitHub 上看不到 CSV 數據文件。

## 根本原因
**data/ 目錄中根本沒有 CSV 文件！**

當前只有：
```
data/DATA_PROVENANCE.md (6.0K)
data/README.md (518 bytes)
```

## 缺失文件清單
❌ `data/supervised_dataset.csv` - 主訓練數據集
❌ `data/toi.csv` - 完整 TOI 數據
❌ `data/toi_positive.csv` - TOI 正樣本
❌ `data/toi_negative.csv` - TOI 負樣本
❌ `data/koi_false_positives.csv` - KOI False Positives

## 為什麼檔案不存在？

### 可能原因 1: 01_tap_download.ipynb 從未執行 (70%)
**症狀**:
- Notebook 存在但從未運行
- 或執行到一半就停止
- 或 Colab session 超時

**驗證方法**:
```python
# 在 Jupyter/Colab 檢查變數是否存在
try:
    print(f"TOI 數據: {len(toi_df)} 筆")
    print(f"合併數據: {len(all_samples)} 筆")
except NameError:
    print("❌ Notebook 從未執行！")
```

### 可能原因 2: 檔案創建在錯誤位置 (20%)
**症狀**:
- 工作目錄錯誤
- 相對路徑 `../data` 指向錯誤位置

**檢查方法**:
```bash
# 搜尋整個系統中的 supervised_dataset.csv
find / -name "supervised_dataset.csv" 2>/dev/null

# Colab 中檢查
!find /content -name "*.csv" 2>/dev/null
```

### 可能原因 3: NASA API 失敗 (10%)
**症狀**:
- API 查詢超時或返回錯誤
- 程式碼繼續執行但數據為空
- 沒有保存文件

**檢查方法**:
查看 notebook 輸出是否有：
```
⚠️ 查詢失敗: ...
⚠️ 生成完整的模擬資料供黑客松使用...
```

## 解決方案

### 方案 A: 重新執行 Notebook（推薦）

1. **在 Google Colab 打開 notebook**:
   ```
   https://colab.research.google.com/github/exoplanet-spaceapps/exoplanet-starter/blob/main/notebooks/01_tap_download.ipynb
   ```

2. **執行所有 cells** (Runtime → Run all)

3. **驗證檔案創建**:
   ```python
   !ls -lh ../data/*.csv
   ```

4. **如果成功，使用改進的推送函數**:
   ```python
   # 使用我提供的 improved_github_push.py
   from improved_github_push import GitHubPushHelper

   helper = GitHubPushHelper()
   helper.run()
   ```

### 方案 B: 使用預生成數據（快速測試）

如果你只是想測試後續流程，可以使用模擬數據：

```python
import pandas as pd
import numpy as np
from pathlib import Path

# 生成模擬數據
np.random.seed(42)
n_samples = 2000

data = {
    'label': np.random.binomial(1, 0.3, n_samples),
    'source': np.random.choice(['TOI_Candidate', 'TOI_FP', 'KOI_FP'], n_samples),
    'period': np.random.lognormal(1.5, 1.0, n_samples),
    'depth': np.random.lognormal(6.5, 1.2, n_samples),
    'duration': np.random.uniform(0.01, 0.5, n_samples),
}

df = pd.DataFrame(data)

# 儲存
Path('../data').mkdir(exist_ok=True)
df.to_csv('../data/supervised_dataset.csv', index=False)

print(f"✅ 生成模擬數據: {len(df)} 筆")
```

### 方案 C: 直接從 NASA 下載（手動）

如果 notebook 執行失敗，可以手動下載：

1. **TOI 數據**:
   - 前往: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI
   - 點擊 "Download Table" → CSV
   - 儲存為 `data/toi.csv`

2. **KOI False Positives**:
   - 前往: https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative
   - 過濾: `koi_disposition = 'FALSE POSITIVE'`
   - 下載為 `data/koi_false_positives.csv`

## Git Push 問題診斷

### 檢查 1: .gitignore 是否排除了 CSV
```bash
grep -E "(data/|\.csv)" .gitignore
```
**結果**: .gitignore 沒有排除 CSV ✅

### 檢查 2: Git LFS 設置
```bash
cat .gitattributes
git lfs ls-files
```
**如果沒有輸出**: LFS 沒有追蹤任何文件

### 檢查 3: 組織 SSO 授權
```bash
git push origin main 2>&1 | grep -i sso
```
**如果有 SSO 錯誤**: Token 未授權給組織

## 改進的推送流程（2025版）

使用我創建的兩個新文件：

### 1. `improved_github_push.py`
完整的 Python 類，處理所有邊緣情況：
- 自動讀取 Colab Secrets
- 檢測組織 SSO
- 正確設置 Git LFS
- 驗證推送成功

### 2. `github_push_cell_2025.py`
可直接複製到 notebook 的 cell：
```python
# 執行新版推送
from github_push_cell_2025 import EnhancedGitHubPush

pusher = EnhancedGitHubPush()
pusher.run_full_push()
```

## 關鍵改進

### 舊版問題:
```python
# ❌ 硬編碼倉庫 URL
repo_url = "https://github.com/exoplanet-spaceapps/exoplanet-starter.git"

# ❌ 工作目錄錯誤
os.chdir('/content')  # 應該是 /content/exoplanet-starter

# ❌ LFS 靜默失敗
subprocess.run(['git', 'lfs', 'track', '*.csv'])  # 不夠完整
```

### 新版改進:
```python
# ✅ 自動檢測倉庫
remote_result = subprocess.run(['git', 'remote', 'get-url', 'origin'])

# ✅ 正確的工作目錄
git_root = self._find_git_root(Path.cwd())

# ✅ 完整 LFS 設置 + 驗證
subprocess.run(['git', 'lfs', 'migrate', 'import', '--include="*.csv"'])
subprocess.run(['git', 'lfs', 'push', auth_url, branch])  # 明確推送
```

## Co-authored-by 格式修正

### 錯誤格式:
```
Co-Authored-By: hctsai1006 <hctsai1006@gmail.com>
```

### 正確格式:
```
Co-authored-by: hctsai1006 <hctsai1006@gmail.com>
```
（`authored` 全小寫）

## 下一步行動

### 立即執行:
1. ✅ 在 Colab 重新執行 `01_tap_download.ipynb`
2. ✅ 驗證 CSV 文件創建: `!ls -lh data/*.csv`
3. ✅ 使用新版推送函數
4. ✅ 檢查 GitHub 上是否有實際文件內容

### 長期改進:
1. 添加數據存在性檢查
2. 在推送前驗證文件
3. 使用 Colab Secrets 儲存 token
4. 設置 GitHub Actions 自動驗證

## 參考資源

- **Git LFS 文檔**: https://git-lfs.github.com/
- **GitHub Organization SSO**: https://docs.github.com/en/authentication/authenticating-with-single-sign-on
- **Colab Secrets**: https://colab.research.google.com/notebooks/secrets.ipynb
- **NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu/

---

**報告生成時間**: 2025-09-30
**診斷版本**: 2025.09.30-ultra