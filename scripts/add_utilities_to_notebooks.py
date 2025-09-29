"""
Script to add reproducibility and logging utilities to all notebooks
"""
# Fix UTF-8 encoding for Windows
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import json
from pathlib import Path

# 設定路徑
project_root = Path(__file__).parent.parent
notebooks_dir = project_root / "notebooks"

# 新的 Cell 4 內容 (插入在環境設定之後，導入套件之前)
NEW_CELL_CONTENT = """# 🔧 設定可重現性與日誌記錄 (2025 Best Practices)
\"\"\"
Phase 1: Critical Infrastructure
- 設定隨機種子確保可重現性
- 初始化日誌記錄系統
- 記錄系統環境資訊
\"\"\"
import sys
import os
from pathlib import Path

# 確保 src 目錄在 Python 路徑中
if IN_COLAB:
    # Colab 環境：專案在 /content/exoplanet-starter
    src_path = Path('/content/exoplanet-starter/src')
else:
    # 本地環境：向上一層找到專案根目錄
    src_path = Path(__file__).parent.parent / 'src' if '__file__' in globals() else Path('../src').resolve()

if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    print(f"📂 已添加 src 路徑: {src_path}")

# 導入工具模組
try:
    from utils import set_random_seeds, setup_logger, get_log_file_path, log_system_info

    # 1️⃣ 設定隨機種子 (確保可重現性)
    set_random_seeds(42)

    # 2️⃣ 設定日誌記錄
    log_file = get_log_file_path("02_bls_baseline", results_dir=Path("../results") if not IN_COLAB else Path("/content/exoplanet-starter/results"))
    logger = setup_logger("02_bls_baseline", log_file=log_file, verbose=True)

    # 3️⃣ 記錄系統資訊
    logger.info("="*60)
    logger.info("🚀 02_bls_baseline.ipynb 開始執行")
    logger.info("="*60)
    log_system_info(logger)

    print("✅ 可重現性與日誌記錄設定完成")
    print(f"   📝 日誌檔案: {log_file}")
    print(f"   🎲 隨機種子: 42")

except ImportError as e:
    print(f"⚠️ 無法導入工具模組: {e}")
    print("   跳過可重現性設定，繼續執行...")

    # 如果導入失敗，創建一個簡單的 logger fallback
    import logging
    logger = logging.getLogger("02_bls_baseline")
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
"""

def insert_cell_after(notebook_path: Path, after_cell_index: int, new_cell_content: str, cell_type: str = "code"):
    """
    在指定 cell 後插入新的 cell

    Args:
        notebook_path: Notebook 檔案路徑
        after_cell_index: 在哪個 cell 之後插入 (0-indexed)
        new_cell_content: 新 cell 的內容
        cell_type: Cell 類型 ("code" 或 "markdown")
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # 建立新 cell
    new_cell = {
        "cell_type": cell_type,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": new_cell_content.split('\n')
    }

    # 插入新 cell
    notebook['cells'].insert(after_cell_index + 1, new_cell)

    # 儲存
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)

    print(f"✅ 已插入新 cell 至 {notebook_path.name} (位置: cell {after_cell_index + 1})")


def check_if_cell_exists(notebook_path: Path, search_text: str) -> bool:
    """檢查 notebook 中是否已存在包含特定文字的 cell"""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    for cell in notebook['cells']:
        cell_source = ''.join(cell.get('source', []))
        if search_text in cell_source:
            return True
    return False


def update_notebook(notebook_name: str, after_cell_index: int = 3):
    """
    更新指定的 notebook，添加可重現性設定

    Args:
        notebook_name: Notebook 檔案名
        after_cell_index: 在哪個 cell 之後插入
    """
    notebook_path = notebooks_dir / notebook_name

    if not notebook_path.exists():
        print(f"❌ 找不到檔案: {notebook_path}")
        return False

    # 檢查是否已經存在相關設定
    if check_if_cell_exists(notebook_path, "set_random_seeds"):
        print(f"⚠️ {notebook_name} 已包含可重現性設定，跳過")
        return False

    print(f"\n📝 更新 {notebook_name}...")

    # 插入新 cell
    insert_cell_after(notebook_path, after_cell_index, NEW_CELL_CONTENT, cell_type="code")

    return True


def main():
    """主函數：更新所有 notebooks"""
    print("🚀 開始更新 notebooks，添加可重現性與日誌記錄...")
    print("="*60)

    # 更新 02_bls_baseline.ipynb (在 Cell 3 之後插入，即 Cell 4 的位置)
    updated = update_notebook("02_bls_baseline.ipynb", after_cell_index=3)

    # 可以繼續更新其他 notebooks
    # update_notebook("03_injection_train.ipynb", after_cell_index=2)
    # update_notebook("04_newdata_inference.ipynb", after_cell_index=2)
    # update_notebook("05_metrics_dashboard.ipynb", after_cell_index=2)

    print("\n" + "="*60)
    if updated:
        print("✅ Notebooks 更新完成！")
        print("\n下一步：")
        print("   1. 在 Colab 或本地環境打開 02_bls_baseline.ipynb")
        print("   2. 執行新增的 Cell 4（可重現性設定）")
        print("   3. 確認日誌記錄正常工作")
    else:
        print("⚠️ 沒有更新任何 notebooks (可能已存在設定)")


if __name__ == "__main__":
    main()