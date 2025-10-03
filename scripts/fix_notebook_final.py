#!/usr/bin/env python3
"""
Final fix for 03_injection_train.ipynb cell order.
Move feature_cols definition (cell 7) before first usage (cell 3).
"""

import json
from pathlib import Path


def main():
    notebook_path = Path("C:/Users/thc1006/Desktop/dev/exoplanet-starter/notebooks/03_injection_train.ipynb")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    print(f"Original cells: {len(cells)}")

    # Find feature_cols definition and usage cells
    def_cell_idx = None
    usage_before_def = []

    for idx, cell in enumerate(cells):
        source = ''.join(cell.get('source', []))
        if 'feature_cols = [' in source or 'feature_cols=[' in source:
            if def_cell_idx is None:
                def_cell_idx = idx
                print(f"Feature definition at cell {idx}")

    for idx, cell in enumerate(cells):
        source = ''.join(cell.get('source', []))
        if 'feature_cols' in source and idx != def_cell_idx:
            if (idx < def_cell_idx and
                any(p in source for p in ['X[feature_cols]', 'df[feature_cols]', '[feature_cols]',
                                          'feature_cols)', 'feature_cols,'])):
                usage_before_def.append(idx)
                print(f"Usage BEFORE definition at cell {idx}")

    if not usage_before_def:
        print("\n[OK] No cells use feature_cols before definition!")
        return

    # Strategy: Move definition cell to position 2 (after imports, before any usage)
    print(f"\nMoving cell {def_cell_idx} to position 2")

    # Extract the definition cell
    def_cell = cells[def_cell_idx]

    # Remove it from current position
    new_cells = cells[:def_cell_idx] + cells[def_cell_idx+1:]

    # Insert at position 2 (after package install and initial imports)
    new_cells = new_cells[:2] + [def_cell] + new_cells[2:]

    # Update notebook
    nb['cells'] = new_cells

    # Reset execution counts
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"[SUCCESS] Saved notebook with {len(new_cells)} cells")

    # Verify
    print("\nVerification:")
    for idx, cell in enumerate(new_cells):
        source = ''.join(cell.get('source', []))
        if 'feature_cols = [' in source or 'feature_cols=[' in source:
            print(f"  Definition now at cell {idx}")
            break

    for idx, cell in enumerate(new_cells):
        source = ''.join(cell.get('source', []))
        if 'feature_cols' in source and 'feature_cols = [' not in source:
            if any(p in source for p in ['X[feature_cols]', 'df[feature_cols]', '[feature_cols]']):
                print(f"  First usage at cell {idx}")
                break

    print("\n[SUCCESS] Notebook fixed and ready for execution!")


if __name__ == "__main__":
    main()