"""
Update all notebooks to use Colab Secrets for GitHub token with fallback
"""
import json
from pathlib import Path

# Old pattern to replace
OLD_TOKEN_CODE = '''    try:
        token = getpass.getpass("請貼上你的 GitHub Token (輸入會被隱藏): ")
        if not token:
            print("❌ Token 不能為空")
            return False
        print("✅ Token 已接收")
    except:
        token = input("請貼上你的 GitHub Token: ")
        if not token:
            print("❌ Token 不能為空")
            return False'''

# New pattern with Colab Secrets
NEW_TOKEN_CODE = '''    # 優先從 Colab Secrets 讀取 GitHub Token
    try:
        from google.colab import userdata
        token = userdata.get('GITHUB_TOKEN')
        print("✅ GitHub Token 已從 Colab Secrets 讀取")
        print("💡 設置方式: Colab 左側欄 🔑 Secrets → 新增 'GITHUB_TOKEN'")
    except:
        # Fallback: 手動輸入
        print("ℹ️  未偵測到 Colab Secrets，請手動輸入 Token")
        try:
            token = getpass.getpass("請貼上你的 GitHub Token (輸入會被隱藏): ")
            if not token:
                print("❌ Token 不能為空")
                return False
            print("✅ Token 已接收")
        except:
            token = input("請貼上你的 GitHub Token: ")
            if not token:
                print("❌ Token 不能為空")
                return False'''

def update_notebook(notebook_path):
    """Update a single notebook"""
    print(f"\n📝 處理: {notebook_path.name}")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified = False

    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        # Convert source to string for processing
        if isinstance(cell['source'], list):
            source_str = ''.join(cell['source'])
        else:
            source_str = cell['source']

        # Check if this cell contains the old token code
        if 'getpass.getpass("請貼上你的 GitHub Token' in source_str:
            print(f"  ✓ 找到 GitHub Token 輸入代碼 (Cell {cell_idx})")

            # Use a more flexible replacement approach
            # Find the try block with getpass and replace it
            import re

            # Pattern to match the getpass token input block
            pattern = r'(    try:\s+token = getpass\.getpass\("請貼上你的 GitHub Token.*?\s+if not token:\s+print\("❌ Token 不能為空"\)\s+return False\s+print\("✅ Token 已接收"\)\s+except:\s+token = input\("請貼上你的 GitHub Token.*?\)\s+if not token:\s+print\("❌ Token 不能為空"\)\s+return False)'

            # Simpler pattern - just find the section and replace
            if 'try:\n        token = getpass.getpass' in source_str:
                # Find start and end of the token input block
                start_marker = 'try:\n        token = getpass.getpass("請貼上你的 GitHub Token'
                end_marker = 'return False'

                start_idx = source_str.find(start_marker)
                if start_idx != -1:
                    # Find the second occurrence of return False after start
                    temp_idx = start_idx
                    for _ in range(2):
                        temp_idx = source_str.find(end_marker, temp_idx + 1)
                        if temp_idx == -1:
                            break

                    if temp_idx != -1:
                        # Extract the old code block
                        end_idx = source_str.find('\n', temp_idx) + 1
                        old_block = source_str[start_idx:end_idx]

                        # Create new block with proper indentation
                        new_block = '''# 優先從 Colab Secrets 讀取 GitHub Token
        try:
            from google.colab import userdata
            token = userdata.get('GITHUB_TOKEN')
            print("✅ GitHub Token 已從 Colab Secrets 讀取")
            print("💡 設置方式: Colab 左側欄 🔑 Secrets → 新增 'GITHUB_TOKEN'")
        except:
            # Fallback: 手動輸入
            print("ℹ️  未偵測到 Colab Secrets，請手動輸入 Token")
            try:
                token = getpass.getpass("請貼上你的 GitHub Token (輸入會被隱藏): ")
                if not token:
                    print("❌ Token 不能為空")
                    return False
                print("✅ Token 已接收")
            except:
                token = input("請貼上你的 GitHub Token: ")
                if not token:
                    print("❌ Token 不能為空")
                    return False
'''

                        new_source_str = source_str[:start_idx] + new_block + source_str[end_idx:]

                        # Convert back to list format (preserve original format)
                        if isinstance(cell['source'], list):
                            cell['source'] = new_source_str.splitlines(keepends=True)
                        else:
                            cell['source'] = new_source_str

                        modified = True
                        print(f"  ✅ 已更新為 Colab Secrets 模式")

    if modified:
        # Write back to file
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"  💾 已保存變更")
        return True
    else:
        print(f"  ℹ️  無需修改")
        return False

def main():
    print("=" * 70)
    print("🔧 更新所有 Notebooks 的 GitHub Token 讀取方式")
    print("=" * 70)
    print("\n📋 變更內容:")
    print("  1️⃣  優先從 Colab Secrets 讀取 GITHUB_TOKEN")
    print("  2️⃣  失敗時自動 fallback 到手動輸入")
    print("  3️⃣  提供清晰的使用說明\n")

    notebooks_dir = Path('notebooks')
    notebooks = list(notebooks_dir.glob('*.ipynb'))

    if not notebooks:
        print("❌ 未找到任何 notebook")
        return

    print(f"📚 找到 {len(notebooks)} 個 notebooks\n")

    modified_count = 0
    for nb_path in sorted(notebooks):
        if update_notebook(nb_path):
            modified_count += 1

    print("\n" + "=" * 70)
    print(f"✅ 完成！共修改 {modified_count} 個 notebooks")
    print("=" * 70)

    if modified_count > 0:
        print("\n💡 使用說明:")
        print("   在 Google Colab 中:")
        print("   1. 點擊左側欄的 🔑 (Secrets) 圖標")
        print("   2. 點擊 '+ Add new secret'")
        print("   3. Name: GITHUB_TOKEN")
        print("   4. Value: 貼上你的 GitHub Personal Access Token")
        print("   5. 啟用 'Notebook access' 開關")
        print("   6. 執行 notebook 時會自動讀取，無需手動輸入！")

if __name__ == "__main__":
    import sys
    import os

    # Ensure UTF-8 output
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8')

    # Change to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)

    main()