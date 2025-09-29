"""Restore notebook to original order - imports should be BEFORE first usage but AFTER data loading"""
import json
import sys

def restore_and_fix_notebook(notebook_path):
    """Find the right place for imports - after data loading, before pipeline creation"""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find key cells
    import_cell_idx = None
    data_load_idx = None
    pipeline_creation_idx = None

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])

            # Import cell
            if 'from utils.gpu_utils import' in source:
                import_cell_idx = i

            # Data loading (look for supervised_dataset.csv or features_df)
            if 'supervised_dataset.csv' in source or ('features_df' in source and '=' in source and 'pd.read' in source):
                if data_load_idx is None:
                    data_load_idx = i

            # Pipeline creation
            if 'create_exoplanet_pipeline' in source and '=' in source:
                if pipeline_creation_idx is None:
                    pipeline_creation_idx = i

    print(f"Import cell: {import_cell_idx}")
    print(f"Data load cell: {data_load_idx}")
    print(f"Pipeline creation cell: {pipeline_creation_idx}")

    if import_cell_idx is None or data_load_idx is None:
        print("ERROR: Could not find required cells")
        return False

    # The imports should come RIGHT BEFORE the first usage (pipeline creation)
    # But AFTER data loading
    correct_position = pipeline_creation_idx - 1 if pipeline_creation_idx else len(nb['cells']) - 1

    if import_cell_idx != correct_position:
        import_cell = nb['cells'].pop(import_cell_idx)
        # Adjust target position if we removed a cell before it
        if import_cell_idx < correct_position:
            correct_position -= 1
        nb['cells'].insert(correct_position, import_cell)
        print(f"SUCCESS: Moved import cell from {import_cell_idx} to {correct_position}")
    else:
        print("SUCCESS: Import cell already in correct position")

    # Save
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"SUCCESS: Fixed notebook saved")
    return True

if __name__ == '__main__':
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else 'notebooks/03_injection_train.ipynb'
    success = restore_and_fix_notebook(notebook_path)
    sys.exit(0 if success else 1)