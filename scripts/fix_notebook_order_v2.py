#!/usr/bin/env python3
"""
Fix cell execution order in 03_injection_train.ipynb - Version 2
Ensures feature_cols is defined BEFORE any usage.
"""

import json
import sys
from pathlib import Path


def main():
    notebook_path = Path("C:/Users/thc1006/Desktop/dev/exoplanet-starter/notebooks/03_injection_train.ipynb")

    if not notebook_path.exists():
        print(f"[ERROR] Notebook not found: {notebook_path}")
        sys.exit(1)

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    print(f"Total cells: {len(cells)}\n")

    # Find feature_cols definition cell
    feature_cols_def_idx = None
    for idx, cell in enumerate(cells):
        source = ''.join(cell.get('source', []))
        if 'feature_cols = [' in source and 'Feature columns' in source:
            feature_cols_def_idx = idx
            print(f"Found feature_cols definition at original cell {idx}")
            break

    if feature_cols_def_idx is None:
        print("[ERROR] Could not find feature_cols definition!")
        sys.exit(1)

    # Find all cells that USE feature_cols
    usage_cells = []
    for idx, cell in enumerate(cells):
        source = ''.join(cell.get('source', []))
        if 'feature_cols' in source and idx != feature_cols_def_idx:
            # Check if it's actual usage (not just a comment)
            if any(pattern in source for pattern in [
                'X[feature_cols]',
                'df[feature_cols]',
                '[feature_cols]',
                'feature_cols)',
                'feature_cols,',
                'in feature_cols',
                'len(feature_cols)'
            ]):
                usage_cells.append(idx)
                print(f"Cell {idx} uses feature_cols")

    # Create new cell order
    new_order = []

    # Phase 1: Keep all cells up to feature_cols definition
    for idx in range(feature_cols_def_idx + 1):
        new_order.append(idx)

    print(f"\nPhase 1 (cells 0-{feature_cols_def_idx}): Setup and feature_cols definition")

    # Phase 2: Add remaining cells that don't use feature_cols
    remaining_cells = []
    for idx in range(feature_cols_def_idx + 1, len(cells)):
        if idx not in usage_cells:
            remaining_cells.append(idx)

    # Add cells that use feature_cols AFTER definition
    usage_cells_after_def = [idx for idx in usage_cells if idx > feature_cols_def_idx]

    print(f"Phase 2: Cells that use feature_cols after definition: {len(usage_cells_after_def)}")
    print(f"Phase 3: Other remaining cells: {len(remaining_cells)}")

    # Build final order: definition → usage cells → other cells
    new_order.extend(usage_cells_after_def)
    new_order.extend(remaining_cells)

    # Handle cells that used feature_cols BEFORE definition (ERROR case)
    problematic_cells = [idx for idx in usage_cells if idx < feature_cols_def_idx]

    if problematic_cells:
        print(f"\n[WARNING] Found {len(problematic_cells)} cells using feature_cols before definition:")
        for idx in problematic_cells:
            print(f"  Cell {idx}")
        print("These cells will be moved AFTER the definition.")

        # Remove problematic cells from beginning and add them after definition
        new_order = [idx for idx in new_order if idx not in problematic_cells]
        # Insert after feature_cols definition
        insert_pos = new_order.index(feature_cols_def_idx) + 1
        for prob_idx in problematic_cells:
            new_order.insert(insert_pos, prob_idx)
            insert_pos += 1

    # Verify all cells are included
    if len(new_order) != len(cells):
        print(f"\n[ERROR] Cell count mismatch! Original: {len(cells)}, New: {len(new_order)}")
        sys.exit(1)

    # Create new notebook
    new_cells = [cells[idx] for idx in new_order]
    nb['cells'] = new_cells

    # Update execution counts
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None  # Reset execution counts

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n[SUCCESS] Saved fixed notebook to: {notebook_path}")

    # Final verification
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    def_pos = new_order.index(feature_cols_def_idx)
    first_usage_pos = min([new_order.index(idx) for idx in usage_cells if idx in new_order], default=-1)

    print(f"feature_cols defined at position: {def_pos}")
    print(f"First usage at position: {first_usage_pos}")

    if def_pos < first_usage_pos:
        print("\n[SUCCESS] Definition comes BEFORE all usage!")
        print(f"Total cells: {len(new_cells)}")
        print("\nNotebook is now ready for sequential execution!")
    else:
        print("\n[ERROR] Definition still comes after usage!")
        sys.exit(1)


if __name__ == "__main__":
    main()