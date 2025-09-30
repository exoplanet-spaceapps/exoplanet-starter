#!/usr/bin/env python
"""Execute notebook cells one by one, removing Colab dependencies."""
import json
import sys
import os

# Set up environment
os.chdir(r'C:\Users\tingy\Desktop\dev\exoplanet-starter')
sys.path.insert(0, r'C:\Users\tingy\Desktop\dev\exoplanet-starter')

def load_notebook(notebook_path):
    """Load notebook and return code cells."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    code_cells = []
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            code_cells.append((i, source))

    return code_cells

def clean_colab_code(source):
    """Remove Colab-specific code."""
    lines = source.split('\n')
    cleaned_lines = []
    skip_block = False

    for line in lines:
        # Skip Colab import blocks
        if 'from google.colab import' in line:
            continue
        if "IN_COLAB = 'google.colab' in sys.modules" in line:
            skip_block = True
            continue
        if skip_block:
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                skip_block = False
            else:
                continue

        # Replace Colab paths
        line = line.replace('/content/drive/MyDrive/', r'C:\Users\tingy\Desktop\dev\exoplanet-starter\data\\')
        line = line.replace('/content/drive/', r'C:\Users\tingy\Desktop\dev\exoplanet-starter\data\\')

        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)

def main():
    notebook_path = r'C:\Users\tingy\Desktop\dev\exoplanet-starter\notebooks\02_bls_baseline.ipynb'
    code_cells = load_notebook(notebook_path)

    print(f"Loaded {len(code_cells)} code cells")
    print("="*80)

    # Print first few cells to understand structure
    for i, (cell_idx, source) in enumerate(code_cells[:5]):
        print(f"\n--- Cell {cell_idx} ---")
        cleaned = clean_colab_code(source)
        print(cleaned[:300] + "..." if len(cleaned) > 300 else cleaned)
        print()

if __name__ == '__main__':
    main()