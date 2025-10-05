#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check download status and suggest next steps"""

from pathlib import Path
import re

h5_files = list(Path('data/lightcurves').glob('*.h5'))
true_count = 0
false_count = 0

for f in h5_files:
    match = re.search(r'SAMPLE_(\d+)', f.stem)
    if match:
        idx = int(match.group(1))
        if idx < 5944:
            true_count += 1
        else:
            false_count += 1

print(f'Total HDF5 files: {len(h5_files)}')
print(f'True samples (0-5943): {true_count}')
print(f'False samples (5944+): {false_count}')
print(f'\nNext steps:')
print(f'1. Extract features from all {len(h5_files)} files')
print(f'2. Prepare balanced dataset: {min(true_count, false_count)*2} samples')
print(f'3. Train XGBoost model')
