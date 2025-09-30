#!/usr/bin/env python
"""Analyze and prepare notebook for local execution."""
import json
import sys

def analyze_notebook(notebook_path):
    """Analyze notebook structure and identify Colab dependencies."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Total cells: {len(nb['cells'])}")
    print(f"Code cells: {sum(1 for c in nb['cells'] if c['cell_type'] == 'code')}")
    print(f"Markdown cells: {sum(1 for c in nb['cells'] if c['cell_type'] == 'markdown')}")
    print("\n" + "="*80)

    colab_cells = []
    path_cells = []

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        source = ''.join(cell.get('source', []))

        # Check for Colab dependencies
        if 'google.colab' in source:
            colab_cells.append(i)
            print(f"\n[Cell {i}] Contains Colab import:")
            print(source[:200] + "..." if len(source) > 200 else source)

        # Check for /content/drive paths
        if '/content/drive' in source:
            path_cells.append(i)
            print(f"\n[Cell {i}] Contains Colab path:")
            print(source[:200] + "..." if len(source) > 200 else source)

    print("\n" + "="*80)
    print(f"Summary:")
    print(f"- Cells with Colab imports: {len(colab_cells)}")
    print(f"- Cells with Colab paths: {len(path_cells)}")

    # List all code cells
    print("\n" + "="*80)
    print("All code cells:")
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            first_line = source.split('\n')[0][:60]
            print(f"Cell {i}: {first_line}...")

if __name__ == '__main__':
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\tingy\Desktop\dev\exoplanet-starter\notebooks\02_bls_baseline.ipynb'
    analyze_notebook(notebook_path)