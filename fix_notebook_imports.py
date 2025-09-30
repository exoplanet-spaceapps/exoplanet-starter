#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fix missing imports in Notebook 02 ENHANCED"""

import json
import sys

def fix_imports():
    # Load notebook
    with open('notebooks/02_bls_baseline_COLAB_ENHANCED.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    fixed_cells = []

    # Fix Cell 10 - add time import
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))

            # Cell with lightkurve and warnings (feature extraction)
            if 'import lightkurve as lk' in source and 'def extract_features_from_lightcurve' in source:
                lines = cell['source']
                new_lines = []
                for line in lines:
                    new_lines.append(line)
                    if 'import warnings' in line and 'import time' not in ''.join(lines):
                        new_lines.append('import time\n')
                        fixed_cells.append(f'Cell {i}: Added import time')
                cell['source'] = new_lines

            # Cell with extract_features_batch function
            if 'def extract_features_batch' in source:
                lines = cell['source']
                source_str = ''.join(lines)
                # Remove old wrong import if exists
                if 'from tqdm.notebook import tqdm' in source_str:
                    lines = [l for l in lines if 'from tqdm.notebook import tqdm' not in l]
                # Add correct import at top
                lines.insert(0, 'from tqdm.notebook import tqdm\n')
                fixed_cells.append(f'Cell {i}: Fixed tqdm import')
                cell['source'] = lines

    # Save
    with open('notebooks/02_bls_baseline_COLAB_ENHANCED.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return fixed_cells

if __name__ == '__main__':
    fixes = fix_imports()
    for fix in fixes:
        print(fix)
    print(f'\n{len(fixes)} fixes applied')