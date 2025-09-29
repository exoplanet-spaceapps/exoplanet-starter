"""Fix notebook import order - add imports at the beginning"""
import json
import sys

def fix_notebook_imports(notebook_path):
    """Move import cells to the beginning of the notebook"""

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the cell with the imports we need
    import_cell_source = """# Phase 3-4 新增導入
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

# 導入自訂模組 (Phase 3)
import sys
import os
if '/content' in os.getcwd():  # Colab 環境
    sys.path.append('/content/exoplanet-starter/src')
else:  # 本地環境
    sys.path.append(os.path.join(os.getcwd(), '..', 'src'))

from models.pipeline import create_exoplanet_pipeline
from utils.gpu_utils import get_xgboost_gpu_params, log_gpu_info

print("✅ Phase 3-4 imports loaded successfully")
print("  - Pipeline, SimpleImputer, RobustScaler")
print("  - StratifiedGroupKFold")
print("  - create_exoplanet_pipeline, get_xgboost_gpu_params")"""

    # Find if this import cell exists
    import_cell_idx = None
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'from utils.gpu_utils import' in source:
                import_cell_idx = i
                break

    if import_cell_idx is None:
        print("ERROR: Import cell not found!")
        return False

    # Move import cell to position 0 (after title)
    if import_cell_idx > 0:
        import_cell = nb['cells'].pop(import_cell_idx)
        nb['cells'].insert(0, import_cell)
        print(f"SUCCESS: Moved import cell from position {import_cell_idx} to position 0")
    else:
        print("SUCCESS: Import cell already at position 0")

    # Save the fixed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"SUCCESS: Fixed notebook saved: {notebook_path}")
    return True

if __name__ == '__main__':
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else 'notebooks/03_injection_train.ipynb'
    success = fix_notebook_imports(notebook_path)
    sys.exit(0 if success else 1)