"""Verify notebook structure"""
import json
import sys
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

notebook_path = Path(__file__).parent.parent / "notebooks" / "02_bls_baseline_COLAB_ENHANCED.ipynb"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print("\nCell structure:")

for i, cell in enumerate(nb['cells']):
    cell_type = cell['cell_type']
    source = cell.get('source', [])
    first_line = source[0][:60] if source else "(empty)"
    # Remove emojis for Windows console
    first_line = first_line.encode('ascii', 'ignore').decode('ascii')
    print(f"{i:2}: {cell_type:8} - {first_line}")

print(f"\n✅ Verification complete!")
print(f"   Expected: 24 cells")
print(f"   Actual: {len(nb['cells'])} cells")
print(f"   Status: {'PASS' if len(nb['cells']) == 24 else 'FAIL'}")