# Google Colab 相容性問題排解指南 (2025年9月版)

## 🚨 主要問題：NumPy 2.0 相容性

Google Colab 於 2025年3月升級至 NumPy 2.0.2，導致多個天文學套件相容性問題。

### 快速診斷
```python
import numpy as np
print(f"NumPy 版本: {np.__version__}")

# 檢查是否為 NumPy 2.0+
if np.__version__.startswith('2.'):
    print("⚠️  檢測到 NumPy 2.0+ - 可能需要特殊處理")
else:
    print("✅ NumPy 1.x - 應該相容")
```

## 📦 套件相容性狀態 (2025年9月)

| 套件 | NumPy 2.0 相容性 | 狀態 | 解決方案 |
|------|------------------|------|----------|
| **lightkurve** | ✅ 良好 | 最新版支援 NumPy 2.0 | 直接安裝 |
| **astroquery** | ⚠️ 部分 | astropy 相依問題 | 用測試版本 |
| **transitleastsquares** | ❌ 不支援 | numba/batman 相依 | 需降版 NumPy |
| **wotan** | ❌ 未知 | 缺乏更新資訊 | 需降版 NumPy |

## 🔧 解決方案

### 方案 A：使用 NumPy 1.26.4 (推薦)
```python
# 在 notebook 第一格執行
!pip install 'numpy<2.0' --force-reinstall
!pip install 'numpy==1.26.4' --force-reinstall

# 重啟運行時（Runtime → Restart runtime）
import numpy as np
print(f"✅ NumPy 版本: {np.__version__}")

# 然後安裝其他套件
!pip install lightkurve astroquery transitleastsquares wotan
```

### 方案 B：分段安裝（可靠性高）
```python
# 第一格：基礎環境
import sys
import subprocess

def safe_install(package):
    """安全安裝套件，包含錯誤處理"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 安裝 {package} 失敗: {e}")
        return False

# 核心依賴
packages_stage1 = [
    'numpy==1.26.4',
    'scipy==1.11.4',
    'matplotlib>=3.7.0',
    'pandas>=2.0.0',
]

print("🔧 安裝第一階段依賴...")
for pkg in packages_stage1:
    safe_install(pkg)

# 需要重啟運行時
print("⚠️  請重啟運行時後繼續執行下一格")
```

```python
# 第二格：天文學套件（重啟後執行）
packages_stage2 = [
    'astropy>=5.3.0',
    'astroquery>=0.4.6',
    'lightkurve>=2.4.0',
]

print("🌌 安裝天文學套件...")
for pkg in packages_stage2:
    safe_install(pkg)
```

```python
# 第三格：進階套件（需要前述依賴）
packages_stage3 = [
    'transitleastsquares',
    'wotan',
    'batman-package',  # TLS 依賴
    'scikit-learn>=1.3.0',
]

print("🔬 安裝進階分析套件...")
for pkg in packages_stage3:
    if not safe_install(pkg):
        print(f"⚠️  {pkg} 安裝失敗，嘗試備選方案...")
        # 備選：從 GitHub 安裝
        if 'transitleastsquares' in pkg:
            safe_install('git+https://github.com/hippke/tls.git')
```

### 方案 C：使用 Colab 回退運行時
```python
# 如果方案 A、B 都失敗，使用官方回退版本
# 在 Colab 設定中選擇 "Fallback runtime"
print("📱 在 Runtime → Change runtime type → Runtime shape → Fallback runtime")
```

## 🧪 相容性測試腳本

將此腳本貼到新 notebook 測試環境：

```python
# 相容性測試套件
def test_compatibility():
    """測試所有關鍵套件的相容性"""
    test_results = {}

    # 測試 NumPy
    try:
        import numpy as np
        test_results['numpy'] = f"✅ {np.__version__}"
    except Exception as e:
        test_results['numpy'] = f"❌ {e}"

    # 測試 Lightkurve
    try:
        import lightkurve as lk
        # 簡單功能測試
        lc = lk.LightCurve(time=[1,2,3], flux=[1,1,1])
        test_results['lightkurve'] = f"✅ {lk.__version__}"
    except Exception as e:
        test_results['lightkurve'] = f"❌ {e}"

    # 測試 Astroquery
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
        test_results['astroquery'] = "✅ 導入成功"
    except Exception as e:
        test_results['astroquery'] = f"❌ {e}"

    # 測試 TLS
    try:
        from transitleastsquares import transitleastsquares
        test_results['transitleastsquares'] = "✅ 導入成功"
    except Exception as e:
        test_results['transitleastsquares'] = f"❌ {e}"

    # 測試 Wotan
    try:
        import wotan
        test_results['wotan'] = "✅ 導入成功"
    except Exception as e:
        test_results['wotan'] = f"❌ {e}"

    # 顯示結果
    print("🧪 相容性測試結果:")
    print("-" * 50)
    for package, result in test_results.items():
        print(f"{package:20} : {result}")

    # 整體評分
    success_count = sum(1 for result in test_results.values() if result.startswith('✅'))
    total_count = len(test_results)
    print("-" * 50)
    print(f"成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")

    return test_results

# 執行測試
test_compatibility()
```

## 🚀 效能最佳化設定

```python
# 記憶體與效能設定
import os
import gc

# 設定環境變數
os.environ['OMP_NUM_THREADS'] = '2'  # 控制並行執行緒
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'

# 記憶體清理函式
def cleanup_memory():
    """清理記憶體避免 OOM"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 GPU 記憶體清理完成 ({torch.cuda.memory_allocated()/1e9:.1f}GB)")
    except ImportError:
        pass
    print("🧹 系統記憶體清理完成")

# 批次處理設定
BATCH_SIZE = 16  # 根據可用記憶體調整
MAX_LIGHTCURVES = 1000  # 每批次最大處理數量
```

## 🆘 常見錯誤與解決方法

### 錯誤 1: "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.0"
```python
# 解決方法：降版 NumPy
!pip install 'numpy<2.0' --force-reinstall
# 重啟運行時必要！
```

### 錯誤 2: "ImportError: cannot import name 'XXX' from 'numpy'"
```python
# 解決方法：確認 NumPy 版本並重新安裝相依套件
!pip install 'numpy==1.26.4' --force-reinstall
!pip uninstall scikit-learn -y
!pip install scikit-learn
```

### 錯誤 3: "RuntimeError: Numpy is not available"
```python
# PyTorch 相容性問題
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 錯誤 4: Lightkurve BLS 互動失效
```python
# 在 Colab 中需要手動啟用 widget
!pip install ipywidgets
from google.colab import output
output.enable_custom_widget_manager()
```

## 📱 聯絡支援

如果以上方法都無效：

1. **檢查 Colab 更新**：https://colab.research.google.com/notebooks/relnotes.ipynb
2. **回報 Bug**：https://github.com/googlecolab/colabtools/issues
3. **社群討論**：https://discuss.ai.google.dev/c/colab
4. **專案問題**：在此 repo 開 Issue

---

*最後更新：2025年9月29日*
*下次計畫更新：根據 NumPy 生態系統發展狀況*