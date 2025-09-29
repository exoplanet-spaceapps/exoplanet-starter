#!/usr/bin/env python3
"""
測試所有 Jupyter notebooks 的基本執行能力
Test basic execution of all Jupyter notebooks
"""
import subprocess
import json
import sys
import io
from pathlib import Path

# Fix Windows Unicode issue
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def test_notebook(notebook_path):
    """測試單個 notebook 的基本執行"""
    print(f"\n📓 Testing: {notebook_path.name}")
    print("=" * 60)

    # 讀取 notebook 內容
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # 檢查基本結構
    if 'cells' not in nb:
        print(f"❌ Invalid notebook structure: no cells found")
        return False

    # 統計 cells
    code_cells = [c for c in nb['cells'] if c['cell_type'] == 'code']
    markdown_cells = [c for c in nb['cells'] if c['cell_type'] == 'markdown']

    print(f"📊 Cells: {len(code_cells)} code, {len(markdown_cells)} markdown")

    # 檢查是否有 NumPy 2.0 修復
    has_numpy_fix = False
    for cell in code_cells:
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'numpy==1.26.4' in source or 'numpy<2.0' in source:
            has_numpy_fix = True
            break

    if has_numpy_fix:
        print("✅ NumPy 2.0 compatibility fix detected")
    else:
        print("⚠️  No NumPy version constraint found")

    # 檢查導入
    imports = set()
    for cell in code_cells:
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'import' in source:
            for line in source.split('\n'):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    imports.add(line.strip().split()[1].split('.')[0])

    if imports:
        print(f"📦 Key imports: {', '.join(sorted(imports)[:5])}")

    return True


def main():
    """測試所有 notebooks"""
    notebooks_dir = Path(__file__).parent.parent / 'notebooks'

    if not notebooks_dir.exists():
        print(f"❌ Notebooks directory not found: {notebooks_dir}")
        return 1

    notebooks = sorted(notebooks_dir.glob('*.ipynb'))

    if not notebooks:
        print(f"❌ No notebooks found in {notebooks_dir}")
        return 1

    print(f"🚀 Found {len(notebooks)} notebooks to test")
    print("=" * 60)

    results = {}
    for nb_path in notebooks:
        try:
            success = test_notebook(nb_path)
            results[nb_path.name] = '✅' if success else '❌'
        except Exception as e:
            print(f"❌ Error testing {nb_path.name}: {e}")
            results[nb_path.name] = '❌'

    # 總結
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print("=" * 60)
    for name, status in results.items():
        print(f"{status} {name}")

    # 重要提醒
    print("\n" + "=" * 60)
    print("⚠️  Important Reminders:")
    print("=" * 60)
    print("1. These are static checks only")
    print("2. Real execution requires Google Colab or local Jupyter")
    print("3. NumPy 2.0 compatibility requires manual runtime restart")
    print("4. GPU features require CUDA-enabled environment")

    return 0 if all(s == '✅' for s in results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())