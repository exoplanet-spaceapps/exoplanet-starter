"""
Comprehensive syntax check for all notebooks
"""
import json
import sys
import ast
import re
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def check_python_syntax(code_str, cell_idx, notebook_name):
    """Check if Python code has syntax errors"""
    errors = []

    # Remove magic commands and shell commands
    lines = code_str.split('\n')
    cleaned_lines = []
    for line in lines:
        # Skip magic commands
        if line.strip().startswith('%') or line.strip().startswith('!'):
            continue
        cleaned_lines.append(line)

    cleaned_code = '\n'.join(cleaned_lines)

    # Check for common problematic patterns
    problematic_patterns = [
        (r'import\s+\w+\s+if\s+.*?\s+else', 'Conditional import statement (invalid syntax)'),
        (r'from\s+\w+\s+import\s+\w+\s+if\s+.*?\s+else', 'Conditional from-import (invalid syntax)'),
    ]

    for pattern, description in problematic_patterns:
        if re.search(pattern, code_str):
            errors.append({
                'type': 'pattern',
                'description': description,
                'line': 'Multiple lines',
                'notebook': notebook_name,
                'cell': cell_idx
            })

    # Try to parse with AST
    if cleaned_code.strip():
        try:
            ast.parse(cleaned_code)
        except SyntaxError as e:
            errors.append({
                'type': 'syntax',
                'description': str(e),
                'line': e.lineno,
                'notebook': notebook_name,
                'cell': cell_idx
            })

    return errors


def check_notebook(notebook_path):
    """Check a single notebook for syntax errors"""
    print(f"\n📓 檢查: {notebook_path.name}")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    all_errors = []

    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        # Get source code
        if isinstance(cell['source'], list):
            code = ''.join(cell['source'])
        else:
            code = cell['source']

        if not code.strip():
            continue

        # Check syntax
        errors = check_python_syntax(code, idx, notebook_path.name)
        all_errors.extend(errors)

    if all_errors:
        print(f"  ❌ 發現 {len(all_errors)} 個問題:")
        for err in all_errors:
            print(f"     • Cell {err['cell']}: {err['description']}")
            if err['line'] != 'Multiple lines':
                print(f"       行號: {err['line']}")
    else:
        print(f"  ✅ 無語法錯誤")

    return all_errors


def main():
    print("=" * 70)
    print("🔍 全面檢查所有 Notebooks 的語法錯誤")
    print("=" * 70)

    notebooks_dir = Path('notebooks')
    notebooks = sorted(notebooks_dir.glob('*.ipynb'))

    if not notebooks:
        print("❌ 未找到任何 notebook")
        return

    print(f"\n📚 找到 {len(notebooks)} 個 notebooks\n")

    total_errors = []

    for nb_path in notebooks:
        errors = check_notebook(nb_path)
        total_errors.extend(errors)

    print("\n" + "=" * 70)
    print("📊 檢查總結")
    print("=" * 70)

    if total_errors:
        print(f"\n❌ 共發現 {len(total_errors)} 個語法問題")
        print("\n按 notebook 分組:")

        errors_by_nb = {}
        for err in total_errors:
            nb_name = err['notebook']
            if nb_name not in errors_by_nb:
                errors_by_nb[nb_name] = []
            errors_by_nb[nb_name].append(err)

        for nb_name, errors in errors_by_nb.items():
            print(f"\n  📓 {nb_name}:")
            for err in errors:
                print(f"     • Cell {err['cell']}: {err['description']}")
    else:
        print("\n✅ 所有 notebooks 均無語法錯誤！")

    return len(total_errors)


if __name__ == "__main__":
    error_count = main()
    sys.exit(0 if error_count == 0 else 1)