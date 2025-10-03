"""
🔧 在 Colab Notebook 03 中插入此 Cell 來修復導入錯誤

在錯誤發生的 Cell 之前插入新 Cell，複製貼上以下代碼：
"""

# ============================================================================
# 🔧 修復 Notebook 03 導入問題
# ============================================================================
import time
import sys
from pathlib import Path

print("=" * 60)
print("🔧 Fixing Notebook 03 imports...")
print("=" * 60)

# 1. 檢查當前環境
print("\n[1/4] Checking environment...")
print(f"   Current directory: {Path.cwd()}")
print(f"   Python version: {sys.version.split()[0]}")

# 2. 設定 Python 路徑
print("\n[2/4] Setting up Python path...")
project_root = Path.cwd()

# 如果在子目錄（如 notebooks/），移到專案根目錄
if project_root.name == 'notebooks':
    project_root = project_root.parent
    print(f"   Detected notebooks directory, using parent: {project_root}")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"   ✅ Added to sys.path: {project_root}")
else:
    print(f"   ✅ Already in sys.path: {project_root}")

# 3. 驗證 app 模組
print("\n[3/4] Verifying app module...")
app_dir = project_root / 'app'
bls_features_file = app_dir / 'bls_features.py'

if app_dir.exists():
    print(f"   ✅ app/ directory found")
    if bls_features_file.exists():
        print(f"   ✅ bls_features.py found")
    else:
        print(f"   ❌ bls_features.py NOT found!")
else:
    print(f"   ❌ app/ directory NOT found!")
    print(f"   Please ensure you're in the project root directory")
    raise FileNotFoundError("app directory not found")

# 4. 導入所有必要函數
print("\n[4/4] Importing functions...")
try:
    from app.bls_features import (
        run_bls,
        extract_features,
        extract_features_batch,
        compute_odd_even_difference,
        compute_transit_symmetry,
        compute_periodicity_strength,
        compute_skewness,
        compute_kurtosis
    )
    print("   ✅ Successfully imported:")
    print("      - run_bls")
    print("      - extract_features")
    print("      - extract_features_batch")
    print("      - compute_odd_even_difference")
    print("      - compute_transit_symmetry")
    print("      - compute_periodicity_strength")
    print("      - compute_skewness")
    print("      - compute_kurtosis")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    print("\n   Troubleshooting:")
    print("   1. Make sure you cloned the full repository")
    print("   2. Run: !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git")
    print("   3. Run: %cd YOUR_REPO")
    print("   4. Re-run this cell")
    raise

# 5. 快速測試
print("\n" + "=" * 60)
print("🧪 Running quick tests...")
print("=" * 60)

# 測試 1: time 模組
print("\n[Test 1] time module:")
print(f"   ✅ Available: {time.__name__}")

# 測試 2: extract_features_batch 函數
print("\n[Test 2] extract_features_batch function:")
import inspect
sig = inspect.signature(extract_features_batch)
print(f"   ✅ Function signature: {sig}")

# 測試 3: 依賴套件
print("\n[Test 3] Dependencies:")
try:
    import pandas as pd
    import numpy as np
    print("   ✅ pandas version:", pd.__version__)
    print("   ✅ numpy version:", np.__version__)

    try:
        import lightkurve as lk
        print("   ✅ lightkurve version:", lk.__version__)
    except ImportError:
        print("   ⚠️  lightkurve not installed (will be installed by notebook)")

except ImportError as e:
    print(f"   ❌ Missing dependency: {e}")

# 完成
print("\n" + "=" * 60)
print("✅ All imports fixed successfully!")
print("=" * 60)
print("\n💡 You can now continue running the notebook cells.")
print("   The error should be resolved.")
print("=" * 60)