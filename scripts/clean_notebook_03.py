"""
Clean Notebook 03 by removing broken cells with undefined dependencies
"""
import json
import re

# Read backup (original) notebook
with open('notebooks/03_injection_train_BACKUP.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(f"🔍 Deep analysis of {len(nb['cells'])} cells...")

# Identify cells by their actual content and purpose
def deep_categorize(cell, idx):
    if cell.get('cell_type') == 'markdown':
        return {
            'index': idx,
            'cell': cell,
            'category': 'markdown',
            'priority': 1000 + idx,
            'keep': True,
            'reason': 'documentation'
        }

    source = ''.join(cell.get('source', []))

    # PHASE 0: Installation (absolute first)
    if 'pip install' in source or '!pip' in source:
        return {
            'index': idx,
            'cell': cell,
            'category': 'install',
            'priority': 0,
            'keep': True,
            'reason': 'package installation'
        }

    # PHASE 1: Core imports
    if (source.strip().startswith('from sklearn.pipeline') or
        source.strip().startswith('from sklearn.preprocessing') or
        source.strip().startswith('from sklearn.model_selection')):
        return {
            'index': idx,
            'cell': cell,
            'category': 'imports_core',
            'priority': 10,
            'keep': True,
            'reason': 'core ML imports'
        }

    # Standard library imports
    if (source.strip().startswith('import json') or
        source.strip().startswith('import time') or
        source.strip().startswith('import numpy') or
        source.strip().startswith('import pandas') or
        source.strip().startswith('from pathlib')):
        return {
            'index': idx,
            'cell': cell,
            'category': 'imports_std',
            'priority': 11,
            'keep': True,
            'reason': 'standard imports'
        }

    # SKIP: Broken calibration cells (reference undefined variables)
    if ('y_pred_uncal' in source or 'xgb_model.predict_proba' in source) and \
       'xgb_model =' not in source and 'def ' not in source:
        return {
            'index': idx,
            'cell': cell,
            'category': 'broken_calibration',
            'priority': 9999,
            'keep': False,
            'reason': 'references undefined xgb_model'
        }

    # SKIP: Cells that use predictions before they exist
    if 'plot_calibration_curves' in source and 'y_pred' in source and 'import' not in source:
        return {
            'index': idx,
            'cell': cell,
            'category': 'broken_viz',
            'priority': 9999,
            'keep': False,
            'reason': 'references undefined predictions'
        }

    # PHASE 2: Data loading
    if 'read_csv' in source and 'supervised' in source:
        return {
            'index': idx,
            'cell': cell,
            'category': 'data_load',
            'priority': 20,
            'keep': True,
            'reason': 'loads training data'
        }

    # PHASE 3: Feature extraction (CRITICAL)
    if 'extract_features_batch' in source or \
       'feature_cols = [col for col' in source:
        return {
            'index': idx,
            'cell': cell,
            'category': 'feature_extraction',
            'priority': 30,
            'keep': True,
            'reason': 'defines feature_cols'
        }

    # PHASE 4: Training preparation
    if 'create_exoplanet_pipeline' in source and 'numerical_features=feature_cols' in source:
        return {
            'index': idx,
            'cell': cell,
            'category': 'training_pipeline',
            'priority': 40,
            'keep': True,
            'reason': 'creates pipeline'
        }

    # PHASE 4b: Cross-validation training
    if 'StratifiedGroupKFold' in source and 'X.iloc[train_idx]' in source:
        return {
            'index': idx,
            'cell': cell,
            'category': 'training_cv',
            'priority': 41,
            'keep': True,
            'reason': 'cross-validation training'
        }

    # PHASE 5: Evaluation (only if valid)
    if ('shap' in source.lower() and 'pipeline' in source) or \
       ('feature_importances_' in source):
        return {
            'index': idx,
            'cell': cell,
            'category': 'evaluation',
            'priority': 50,
            'keep': True,
            'reason': 'model evaluation'
        }

    # PHASE 6: Saving
    if 'joblib.dump' in source or ('to_csv' in source and 'results' in source):
        return {
            'index': idx,
            'cell': cell,
            'category': 'saving',
            'priority': 60,
            'keep': True,
            'reason': 'save outputs'
        }

    # GitHub push cells
    if 'ultimate_push_to_github' in source or 'GitHub Push' in source:
        return {
            'index': idx,
            'cell': cell,
            'category': 'github',
            'priority': 70,
            'keep': True,
            'reason': 'github integration'
        }

    # Default: misc code (keep but deprioritize)
    return {
        'index': idx,
        'cell': cell,
        'category': 'misc',
        'priority': 55,
        'keep': True,
        'reason': 'miscellaneous code'
    }

# Categorize all cells
categorized = []
for idx, cell in enumerate(nb['cells']):
    cat = deep_categorize(cell, idx)
    categorized.append(cat)

# Filter out broken cells
kept_cells = [c for c in categorized if c['keep']]
removed_cells = [c for c in categorized if not c['keep']]

print(f"\n📊 Analysis Results:")
print(f"   ✅ Keeping: {len(kept_cells)} cells")
print(f"   ❌ Removing: {len(removed_cells)} broken cells")

for removed in removed_cells:
    print(f"      Cell {removed['index']}: {removed['reason']}")

# Sort kept cells by priority
kept_cells_sorted = sorted(kept_cells, key=lambda x: (x['priority'], x['index']))

# Create clean notebook
nb_clean = nb.copy()
nb_clean['cells'] = [c['cell'] for c in kept_cells_sorted]

# Final verification
print(f"\n🔍 Final Verification:")
feature_cols_def = None
feature_cols_use = None

for new_idx, info in enumerate(kept_cells_sorted):
    if info['category'] != 'markdown':
        source = ''.join(info['cell'].get('source', []))

        if 'feature_cols = [col' in source and feature_cols_def is None:
            feature_cols_def = new_idx
            print(f"   ✅ feature_cols defined at cell {new_idx}")

        if 'numerical_features=feature_cols' in source and feature_cols_use is None:
            feature_cols_use = new_idx
            print(f"   📍 feature_cols first used at cell {new_idx}")

if feature_cols_def and feature_cols_use:
    if feature_cols_def < feature_cols_use:
        print(f"   ✅ VALID ORDER (gap: {feature_cols_use - feature_cols_def} cells)")
    else:
        print(f"   ❌ INVALID ORDER!")

# Save clean notebook
output_path = 'notebooks/03_injection_train.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(nb_clean, f, ensure_ascii=False, indent=1)

print(f"\n💾 Saved clean notebook:")
print(f"   Path: {output_path}")
print(f"   Total cells: {len(nb_clean['cells'])}")
print(f"   Removed broken cells: {len(removed_cells)}")
print(f"\n✅ Ready for execution!")