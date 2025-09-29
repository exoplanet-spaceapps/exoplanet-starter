"""Swap cells in notebook to fix execution order"""
import json
import sys

def swap_cells(notebook_path, cell1_idx, cell2_idx):
    """Swap two cells in the notebook"""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    if cell1_idx >= len(nb['cells']) or cell2_idx >= len(nb['cells']):
        print(f"ERROR: Invalid cell indices")
        return False

    # Swap
    nb['cells'][cell1_idx], nb['cells'][cell2_idx] = nb['cells'][cell2_idx], nb['cells'][cell1_idx]

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"SUCCESS: Swapped cells {cell1_idx} and {cell2_idx}")
    return True

if __name__ == '__main__':
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else 'notebooks/03_injection_train.ipynb'
    cell1 = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    cell2 = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    success = swap_cells(notebook_path, cell1, cell2)
    sys.exit(0 if success else 1)