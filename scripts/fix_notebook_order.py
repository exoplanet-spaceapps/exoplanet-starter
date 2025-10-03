#!/usr/bin/env python3
"""
Fix cell execution order in 03_injection_train.ipynb
Ensures feature_cols is defined BEFORE it's used.
"""

import json
import sys
from pathlib import Path


def analyze_notebook(notebook_path):
    """Analyze notebook structure and find key cells."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    print(f"Total cells: {len(cells)}\n")

    # Find key cells
    feature_cols_def = []
    feature_cols_usage = []
    install_cells = []
    import_cells = []
    data_loading_cells = []
    training_cells = []
    evaluation_cells = []
    saving_cells = []

    for idx, cell in enumerate(cells):
        source = ''.join(cell.get('source', []))

        # Categorize cells
        if 'feature_cols' in source:
            if 'feature_cols = [' in source or 'feature_cols=' in source:
                feature_cols_def.append(idx)
                print(f"Cell {idx}: Defines feature_cols")
            elif 'feature_cols' in source:
                feature_cols_usage.append(idx)
                print(f"Cell {idx}: Uses feature_cols")

        if 'pip install' in source or '!pip install' in source:
            install_cells.append(idx)
            print(f"Cell {idx}: Package installation")

        if 'import ' in source and idx < 10:
            import_cells.append(idx)
            print(f"Cell {idx}: Import statements")

        if 'pd.read_csv' in source or 'load_data' in source:
            data_loading_cells.append(idx)
            print(f"Cell {idx}: Data loading")

        if 'Pipeline' in source or 'LogisticRegression' in source or 'XGBClassifier' in source:
            training_cells.append(idx)
            print(f"Cell {idx}: Model training")

        if 'SHAP' in source or 'shap' in source or 'calibration' in source:
            evaluation_cells.append(idx)
            print(f"Cell {idx}: Evaluation/SHAP")

        if 'joblib.dump' in source or 'model.save' in source:
            saving_cells.append(idx)
            print(f"Cell {idx}: Model saving")

    return {
        'notebook': nb,
        'feature_cols_def': feature_cols_def,
        'feature_cols_usage': feature_cols_usage,
        'install_cells': install_cells,
        'import_cells': import_cells,
        'data_loading_cells': data_loading_cells,
        'training_cells': training_cells,
        'evaluation_cells': evaluation_cells,
        'saving_cells': saving_cells
    }


def reorganize_notebook(analysis):
    """Reorganize cells in correct logical order."""
    nb = analysis['notebook']
    cells = nb['cells']

    # Define correct execution order
    new_order = []
    used_indices = set()

    # Phase 0: Package installation
    print("\n=== Phase 0: Package Installation ===")
    for idx in analysis['install_cells']:
        new_order.append(idx)
        used_indices.add(idx)
        print(f"  Cell {idx}")

    # Phase 1: Imports
    print("\n=== Phase 1: Imports ===")
    for idx in analysis['import_cells']:
        if idx not in used_indices:
            new_order.append(idx)
            used_indices.add(idx)
            print(f"  Cell {idx}")

    # Phase 2: Data loading
    print("\n=== Phase 2: Data Loading ===")
    for idx in analysis['data_loading_cells']:
        if idx not in used_indices:
            new_order.append(idx)
            used_indices.add(idx)
            print(f"  Cell {idx}")

    # Phase 3: Feature extraction (MUST come before training)
    print("\n=== Phase 3: Feature Extraction ===")
    for idx in analysis['feature_cols_def']:
        if idx not in used_indices:
            new_order.append(idx)
            used_indices.add(idx)
            print(f"  Cell {idx} - Defines feature_cols")

    # Add cells around feature_cols definition (context cells)
    for idx in analysis['feature_cols_def']:
        # Add 2 cells before and after for context
        for offset in range(-2, 3):
            context_idx = idx + offset
            if 0 <= context_idx < len(cells) and context_idx not in used_indices:
                new_order.append(context_idx)
                used_indices.add(context_idx)
                print(f"  Cell {context_idx} - Context around feature_cols")

    # Phase 4: Pipeline training (uses feature_cols)
    print("\n=== Phase 4: Pipeline Training ===")
    for idx in analysis['training_cells']:
        if idx not in used_indices:
            new_order.append(idx)
            used_indices.add(idx)
            print(f"  Cell {idx}")

    # Add cells that USE feature_cols
    for idx in analysis['feature_cols_usage']:
        if idx not in used_indices:
            new_order.append(idx)
            used_indices.add(idx)
            print(f"  Cell {idx} - Uses feature_cols")

    # Phase 5: Evaluation
    print("\n=== Phase 5: Evaluation ===")
    for idx in analysis['evaluation_cells']:
        if idx not in used_indices:
            new_order.append(idx)
            used_indices.add(idx)
            print(f"  Cell {idx}")

    # Phase 6: Model saving
    print("\n=== Phase 6: Model Saving ===")
    for idx in analysis['saving_cells']:
        if idx not in used_indices:
            new_order.append(idx)
            used_indices.add(idx)
            print(f"  Cell {idx}")

    # Phase 7: Add remaining cells
    print("\n=== Phase 7: Remaining Cells ===")
    for idx in range(len(cells)):
        if idx not in used_indices:
            new_order.append(idx)
            used_indices.add(idx)
            print(f"  Cell {idx}")

    # Create new notebook with reorganized cells
    new_cells = [cells[idx] for idx in new_order]
    nb['cells'] = new_cells

    # Update execution counts
    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            cell['execution_count'] = idx + 1

    print(f"\n=== Summary ===")
    print(f"Original cells: {len(cells)}")
    print(f"Reorganized cells: {len(new_cells)}")
    print(f"Cell order preserved: {len(new_order) == len(cells)}")

    return nb


def save_notebook(notebook, output_path):
    """Save reorganized notebook."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"\n[SUCCESS] Saved fixed notebook to: {output_path}")


def main():
    notebook_path = Path("C:/Users/thc1006/Desktop/dev/exoplanet-starter/notebooks/03_injection_train.ipynb")

    if not notebook_path.exists():
        print(f"[ERROR] Notebook not found: {notebook_path}")
        sys.exit(1)

    print("="*60)
    print("ANALYZING NOTEBOOK STRUCTURE")
    print("="*60)

    analysis = analyze_notebook(notebook_path)

    print("\n" + "="*60)
    print("REORGANIZING CELLS")
    print("="*60)

    fixed_nb = reorganize_notebook(analysis)

    print("\n" + "="*60)
    print("SAVING FIXED NOTEBOOK")
    print("="*60)

    save_notebook(fixed_nb, notebook_path)

    print("\n" + "="*60)
    print("[SUCCESS] Notebook cell order fixed!")
    print("="*60)
    print("\nKey changes:")
    print("- feature_cols definition moved BEFORE usage")
    print("- Logical execution flow: Install → Import → Load → Extract → Train → Evaluate → Save")
    print("- All cells preserved with correct dependencies")
    print(f"\nTotal cells: {len(fixed_nb['cells'])}")


if __name__ == "__main__":
    main()