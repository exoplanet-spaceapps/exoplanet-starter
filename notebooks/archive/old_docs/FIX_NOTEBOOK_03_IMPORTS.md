# 修復 Notebook 03 導入錯誤

## 問題
執行 `03_injection_train.ipynb` 時遇到兩個錯誤：
1. `NameError: name 'extract_features_batch' is not defined`
2. `time` 模組未導入

## 解決方案

在 Colab 執行 Notebook 03 時，**在錯誤的 Cell 之前**插入以下修復代碼：

### 修復方案 1：直接在 Notebook 中添加導入（推薦）

在 Notebook 03 的**第一個 Cell**（或錯誤發生的 Cell 之前）添加：

```python
# ============================================================================
# 🔧 修復導入問題
# ============================================================================
import time
import sys
from pathlib import Path

# 確保可以導入 app 模組
if '/content' in str(Path.cwd()):
    # Colab 環境
    sys.path.insert(0, '/content')
else:
    # 本地環境
    sys.path.insert(0, str(Path.cwd().parent))

# 導入所有必要的函數
from app.bls_features import (
    run_bls,
    extract_features,
    extract_features_batch,
    compute_odd_even_difference,
    compute_transit_symmetry,
    compute_periodicity_strength
)

print("✅ All imports successful!")
print(f"   - time module: {time}")
print(f"   - extract_features_batch: {extract_features_batch}")
```

### 修復方案 2：更新現有的導入 Cell

找到 Notebook 中的這個 Cell（大約在 Cell 4-6）：

```python
from app.bls_features import (
    run_bls,
    extract_features_batch,
    # ...
)
```

**確認：**
1. ✅ `extract_features_batch` 在導入列表中
2. ✅ 在此 Cell **之前**有 `import time`
3. ✅ `sys.path` 設定正確

如果缺少，在該 Cell **頂部**添加：

```python
import time
import sys
from pathlib import Path

# 設定 Python 路徑
sys.path.insert(0, str(Path.cwd().parent))
```

### 修復方案 3：檢查 Colab 工作目錄

如果上述方法仍失敗，在 Notebook 最開始執行：

```python
# 檢查當前環境
import os
import sys
from pathlib import Path

print("Current working directory:", os.getcwd())
print("Python path:", sys.path[:3])

# 列出 app 目錄內容
app_dir = Path('app')
if app_dir.exists():
    print("\n✅ app/ directory found:")
    print("   Files:", list(app_dir.glob('*.py')))
else:
    print("\n❌ app/ directory NOT found!")
    print("   Trying parent directory...")
    app_dir = Path('../app')
    if app_dir.exists():
        print("   ✅ Found at ../app/")
        sys.path.insert(0, str(Path.cwd().parent))
    else:
        print("   ❌ app/ not found in parent either!")
        print("   You may need to clone the repo properly in Colab")

# 驗證導入
try:
    from app.bls_features import extract_features_batch
    print("\n✅ extract_features_batch imported successfully!")
except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you cloned the full repository in Colab")
    print("2. Run: !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git")
    print("3. Run: %cd YOUR_REPO")
```

## 快速測試

執行以下代碼確認修復成功：

```python
# 測試 1: time 模組
try:
    import time
    print("✅ time module imported")
except ImportError:
    print("❌ time module import failed")

# 測試 2: extract_features_batch
try:
    from app.bls_features import extract_features_batch
    print("✅ extract_features_batch imported")
except ImportError as e:
    print(f"❌ extract_features_batch import failed: {e}")

# 測試 3: 函數可用性
try:
    import pandas as pd
    import numpy as np

    # 創建測試資料
    test_df = pd.DataFrame({
        'sample_id': ['test_1'],
        'time': [np.array([1, 2, 3, 4, 5])],
        'flux': [np.array([1.0, 0.99, 1.0, 0.99, 1.0])]
    })

    # 測試函數
    result = extract_features_batch(test_df, verbose=False)
    print("✅ extract_features_batch function works!")
    print(f"   Result shape: {result.shape}")
except Exception as e:
    print(f"❌ Function test failed: {e}")
```

## 預期輸出

如果修復成功，應該看到：

```
✅ All imports successful!
   - time module: <module 'time' (built-in)>
   - extract_features_batch: <function extract_features_batch at 0x...>
✅ time module imported
✅ extract_features_batch imported
✅ extract_features_batch function works!
   Result shape: (1, 15)
```

## 在 Colab 執行的完整修復步驟

### 步驟 1：確保專案結構正確

```python
# 在 Colab 的第一個 Cell 執行
!git clone https://github.com/YOUR_USERNAME/exoplanet-starter.git
%cd exoplanet-starter
!ls -la app/
```

應該看到 `app/bls_features.py` 存在。

### 步驟 2：添加修復 Cell

在現有的導入 Cell **之前**插入新 Cell：

```python
# ============================================================================
# 🔧 環境設定與導入修復
# ============================================================================
import time
import sys
from pathlib import Path

# 確保專案根目錄在 Python 路徑中
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 驗證 app 模組可訪問
try:
    import app.bls_features
    print("✅ app.bls_features module found")
except ImportError as e:
    print(f"❌ Cannot import app.bls_features: {e}")
    print("Current directory:", Path.cwd())
    print("sys.path:", sys.path[:3])
    raise

# 導入所有必要函數
from app.bls_features import (
    run_bls,
    extract_features,
    extract_features_batch
)

print("✅ All imports successful!")
```

### 步驟 3：執行 Notebook

現在執行所有 Cell，錯誤應該已解決。

## 替代方案：使用修復過的 Notebook 版本

如果上述方法太複雜，可以使用 `03_injection_train_MINIMAL.ipynb`，這個版本：
- ✅ 已修復導入問題
- ✅ 代碼更簡潔
- ✅ 適合快速測試

在 Colab 開啟：
```
https://colab.research.google.com/github/YOUR_REPO/blob/main/notebooks/03_injection_train_MINIMAL.ipynb
```

## 常見問題

### Q: 為什麼會出現這個錯誤？
A: Notebook 03 的導入 Cell 順序可能不正確，或者在 Colab 環境中 `sys.path` 設定不當。

### Q: 我應該修改本地檔案還是在 Colab 修改？
A: **在 Colab 中插入修復 Cell** 最快。如果要永久修復，可以更新 GitHub 上的 Notebook。

### Q: 修復後性能會受影響嗎？
A: 不會，這只是導入修復，不影響訓練性能。

## 驗證修復成功

修復後，執行以下代碼確認：

```python
# 完整驗證
import inspect

print("=== Verification Report ===\n")

# 1. time module
print("1. time module:")
print(f"   Available: {time is not None}")
print(f"   Functions: {dir(time)[:5]}")

# 2. extract_features_batch
print("\n2. extract_features_batch:")
print(f"   Available: {extract_features_batch is not None}")
print(f"   Signature: {inspect.signature(extract_features_batch)}")

# 3. Dependencies
print("\n3. Dependencies:")
try:
    import pandas as pd
    import numpy as np
    import lightkurve as lk
    print("   ✅ pandas, numpy, lightkurve all available")
except ImportError as e:
    print(f"   ❌ Missing dependency: {e}")

print("\n✅ Verification complete!")
```

## 需要協助？

如果仍有問題，請提供：
1. 完整的錯誤訊息
2. 當前工作目錄（`!pwd`）
3. Python 路徑（`print(sys.path)`）
4. app 目錄內容（`!ls -la app/`）