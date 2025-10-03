#!/usr/bin/env python3
"""
Complete Notebook Formatting Fix
================================

Comprehensive fix for all formatting issues in the notebook.
"""

import nbformat
import re
from pathlib import Path
import shutil


def fix_code_formatting(source: str) -> str:
    """Fix all code formatting issues."""
    lines = []

    # First, split by actual newlines
    raw_lines = source.split('\n')

    for line in raw_lines:
        # Remove trailing whitespace
        line = line.rstrip()

        # Skip empty lines
        if not line:
            lines.append('')
            continue

        # Fix statements that should be on separate lines
        # Pattern 1: try:    import (with multiple spaces)
        if re.match(r'^(\s*)try:\s+import', line):
            indent = re.match(r'^(\s*)', line).group(1)
            lines.append(f"{indent}try:")
            rest = re.sub(r'^(\s*)try:\s+', '', line)
            lines.append(f"{indent}    {rest}")
            continue

        # Pattern 2: except:    ... (with multiple spaces)
        if re.match(r'^(\s*)except:\s+', line):
            indent = re.match(r'^(\s*)', line).group(1)
            lines.append(f"{indent}except:")
            rest = re.sub(r'^(\s*)except:\s+', '', line)
            lines.append(f"{indent}    {rest}")
            continue

        # Pattern 3: if ...:    ... (condition on one line with body)
        if_match = re.match(r'^(\s*)(if .+?):\s{2,}(.+)$', line)
        if if_match:
            indent, condition, body = if_match.groups()
            lines.append(f"{indent}{condition}:")
            lines.append(f"{indent}    {body}")
            continue

        # Pattern 4: else:    ... (with multiple spaces)
        else_match = re.match(r'^(\s*)(else):\s{2,}(.+)$', line)
        if else_match:
            indent, keyword, body = else_match.groups()
            lines.append(f"{indent}{keyword}:")
            lines.append(f"{indent}    {body}")
            continue

        # Default: keep line as is
        lines.append(line)

    return '\n'.join(lines)


def fix_markdown_formatting(source: str) -> str:
    """Fix markdown formatting issues."""
    lines = []

    for line in source.split('\n'):
        # Remove trailing whitespace
        line = line.rstrip()

        # Fix markdown headers without space after #
        header_match = re.match(r'^(#{1,6})([^#\s].*)', line)
        if header_match:
            hashes, content = header_match.groups()
            line = f"{hashes} {content}"

        lines.append(line)

    return '\n'.join(lines)


def process_notebook(input_path: Path, output_path: Path = None, create_backup: bool = True) -> dict:
    """Process and fix notebook formatting."""
    if output_path is None:
        output_path = input_path

    # Create backup
    if create_backup and output_path == input_path:
        backup_path = input_path.parent / f"{input_path.stem}_backup{input_path.suffix}"
        shutil.copy2(input_path, backup_path)
        print(f"Backup created: {backup_path}")

    # Read notebook
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    stats = {
        'total_cells': len(nb.cells),
        'code_cells': 0,
        'markdown_cells': 0,
        'cells_modified': 0
    }

    # Process each cell
    for i, cell in enumerate(nb.cells):
        original_source = cell.source

        if cell.cell_type == 'code':
            stats['code_cells'] += 1
            cell.source = fix_code_formatting(original_source)

            # Clear outputs
            cell.outputs = []
            cell.execution_count = None

        elif cell.cell_type == 'markdown':
            stats['markdown_cells'] += 1
            cell.source = fix_markdown_formatting(original_source)

        if cell.source != original_source:
            stats['cells_modified'] += 1

    # Write fixed notebook
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    return stats


def main():
    """Main execution."""
    input_path = Path('notebooks/03_injection_train_FIXED.ipynb')

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return

    print("=" * 70)
    print("Complete Notebook Formatting Fix")
    print("=" * 70)
    print()

    print(f"Processing: {input_path}")
    print()

    # Process notebook
    stats = process_notebook(input_path, create_backup=True)

    print()
    print("Results:")
    print(f"  Total cells: {stats['total_cells']}")
    print(f"  Code cells: {stats['code_cells']}")
    print(f"  Markdown cells: {stats['markdown_cells']}")
    print(f"  Cells modified: {stats['cells_modified']}")
    print()

    print("Fixes applied:")
    print("  [OK] Fixed try/except statements on one line")
    print("  [OK] Fixed if/else statements on one line")
    print("  [OK] Fixed markdown header spacing")
    print("  [OK] Removed trailing whitespace")
    print("  [OK] Cleared all cell outputs")
    print()

    print("=" * 70)
    print("Done! Notebook formatting fixed successfully.")
    print("=" * 70)


if __name__ == '__main__':
    main()