#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Count final HDF5 files by True/False"""

from pathlib import Path
import h5py

h5_files = list(Path('data/lightcurves').glob('*.h5'))
true_count = 0
false_count = 0

for f in h5_files:
    try:
        with h5py.File(f, 'r') as hf:
            sample_id = hf.attrs.get('sample_id', '')
            if 'SAMPLE_' in str(sample_id):
                idx = int(str(sample_id).split('_')[1])
                if idx < 5944:
                    true_count += 1
                else:
                    false_count += 1
    except Exception as e:
        print(f"Error reading {f.name}: {e}")

print(f"Total HDF5 files: {len(h5_files)}")
print(f"True samples (index 0-5943): {true_count}")
print(f"False samples (index 5944+): {false_count}")
print(f"\nBalanced training set: {min(true_count, false_count)*2} samples")
print(f"  -> {min(true_count, false_count)} True + {min(true_count, false_count)} False")
