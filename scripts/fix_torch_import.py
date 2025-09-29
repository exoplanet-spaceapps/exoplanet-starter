"""
Fix torch import syntax error in 02_bls_baseline.ipynb
"""
import json
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Read notebook
with open('notebooks/02_bls_baseline.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and fix Cell 3 (index 3) - the environment setup cell
cell = nb['cells'][3]

print(f"Current cell type: {cell['cell_type']}")

# Find and replace the problematic line
if isinstance(cell['source'], list):
    new_source = []
    skip_next = 0

    for i, line in enumerate(cell['source']):
        if skip_next > 0:
            skip_next -= 1
            continue

        if 'import torch if' in line:
            print(f"Found problematic line at index {i}: {line[:50]}...")

            # Replace with correct syntax
            new_source.append('# 檢查 GPU 資訊（嘗試導入 torch）\n')
            new_source.append('try:\n')
            new_source.append('    import torch\n')
            new_source.append('except ImportError:\n')
            new_source.append('    torch = None\n')
            new_source.append('\n')

            # Skip the original problematic if statement (next line)
            skip_next = 1
        else:
            new_source.append(line)

    cell['source'] = new_source

# Save
with open('notebooks/02_bls_baseline.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('✅ 已修復 Cell 3 的 torch import 語法錯誤')
print('   原本: import torch if \'torch\' in [m.name for m in pkgutil.iter_modules()] else None')
print('   改為: try-except 區塊')