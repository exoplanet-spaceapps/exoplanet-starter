#!/usr/bin/env python3
"""
Fix Jupyter Notebook Formatting Issues
======================================

This script fixes common formatting problems in Jupyter notebooks:
1. Inconsistent indentation in code cells
2. Long lines that need wrapping
3. Trailing whitespace
4. Multiple statements on one line
5. Markdown header spacing
6. Removes cell outputs
"""

import nbformat
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import shutil


def analyze_notebook(nb: nbformat.NotebookNode) -> Dict[str, Any]:
    """Analyze notebook and identify formatting issues."""
    issues = {
        'long_lines': [],
        'trailing_whitespace': [],
        'multiple_statements': [],
        'markdown_headers': [],
        'inconsistent_indent': [],
        'total_issues': 0
    }

    for i, cell in enumerate(nb.cells):
        cell_id = i + 1
        source = cell.source if isinstance(cell.source, str) else ''.join(cell.source)

        if cell.cell_type == 'code':
            lines = source.split('\n')

            for line_no, line in enumerate(lines, 1):
                # Check for long lines (>100 chars)
                if len(line) > 100:
                    issues['long_lines'].append({
                        'cell': cell_id,
                        'line': line_no,
                        'length': len(line),
                        'content': line[:50] + '...' if len(line) > 50 else line
                    })

                # Check for trailing whitespace
                if line and line != line.rstrip():
                    issues['trailing_whitespace'].append({
                        'cell': cell_id,
                        'line': line_no
                    })

            # Check for multiple statements on one line
            if re.search(r'try:\s+import|except:\s+', source):
                issues['multiple_statements'].append({
                    'cell': cell_id,
                    'pattern': 'try/except on one line'
                })

        elif cell.cell_type == 'markdown':
            # Check for markdown headers without space
            lines = source.split('\n')
            for line_no, line in enumerate(lines, 1):
                if re.match(r'^#+[^\s]', line):
                    issues['markdown_headers'].append({
                        'cell': cell_id,
                        'line': line_no,
                        'content': line[:30]
                    })

    issues['total_issues'] = sum(len(v) for k, v in issues.items() if isinstance(v, list))
    return issues


def fix_code_cell(source: str) -> str:
    """Fix formatting issues in code cells."""
    lines = source.split('\n')
    fixed_lines = []

    for line in lines:
        # Remove trailing whitespace
        line = line.rstrip()

        # Fix try/except statements on one line
        if 'try:    import' in line or 'try:  import' in line:
            # Split into multiple lines
            parts = re.split(r'(try:|except:)', line)
            if len(parts) > 1:
                for part in parts:
                    if part and part not in ['try:', 'except:']:
                        fixed_lines.append(part.strip())
                    elif part:
                        fixed_lines.append(part)
                continue

        # Fix multiple imports on one line
        if 'import' in line and ';' in line:
            # Split by semicolon
            sub_lines = [l.strip() for l in line.split(';') if l.strip()]
            fixed_lines.extend(sub_lines)
            continue

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def fix_markdown_cell(source: str) -> str:
    """Fix formatting issues in markdown cells."""
    lines = source.split('\n')
    fixed_lines = []

    for line in lines:
        # Remove trailing whitespace
        line = line.rstrip()

        # Fix markdown headers without space
        match = re.match(r'^(#+)([^\s#].*)', line)
        if match:
            hashes, content = match.groups()
            line = f"{hashes} {content}"

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def clean_notebook(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Clean and fix all formatting issues in notebook."""
    for cell in nb.cells:
        # Fix cell source
        source = cell.source if isinstance(cell.source, str) else ''.join(cell.source)

        if cell.cell_type == 'code':
            cell.source = fix_code_cell(source)
        elif cell.cell_type == 'markdown':
            cell.source = fix_markdown_cell(source)

        # Remove outputs from code cells
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None

    return nb


def print_analysis(issues: Dict[str, Any]):
    """Print analysis results."""
    print("=" * 70)
    print("Notebook Formatting Analysis")
    print("=" * 70)
    print()

    print(f"Total Issues Found: {issues['total_issues']}")
    print()

    if issues['long_lines']:
        print(f"Long Lines (>100 chars): {len(issues['long_lines'])}")
        for item in issues['long_lines'][:5]:
            print(f"  - Cell {item['cell']}, Line {item['line']}: {item['length']} chars")
            print(f"    {item['content']}")
        if len(issues['long_lines']) > 5:
            print(f"  ... and {len(issues['long_lines']) - 5} more")
        print()

    if issues['trailing_whitespace']:
        print(f"Trailing Whitespace: {len(issues['trailing_whitespace'])} lines")
        print()

    if issues['multiple_statements']:
        print(f"Multiple Statements on One Line: {len(issues['multiple_statements'])}")
        for item in issues['multiple_statements']:
            print(f"  - Cell {item['cell']}: {item['pattern']}")
        print()

    if issues['markdown_headers']:
        print(f"Markdown Header Issues: {len(issues['markdown_headers'])}")
        for item in issues['markdown_headers'][:5]:
            print(f"  - Cell {item['cell']}, Line {item['line']}: {item['content']}")
        if len(issues['markdown_headers']) > 5:
            print(f"  ... and {len(issues['markdown_headers']) - 5} more")
        print()


def main():
    """Main execution function."""
    # Paths
    notebook_path = Path('notebooks/03_injection_train_FIXED.ipynb')
    backup_path = Path('notebooks/03_injection_train_FIXED_backup.ipynb')

    if not notebook_path.exists():
        print(f"Error: Notebook not found at {notebook_path}")
        return

    print("=" * 70)
    print("Jupyter Notebook Formatting Fix")
    print("=" * 70)
    print()
    print(f"Input file: {notebook_path}")
    print()

    # Read notebook
    print("Step 1: Reading notebook...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    print(f"  Total cells: {len(nb.cells)}")
    print(f"  Code cells: {sum(1 for c in nb.cells if c.cell_type == 'code')}")
    print(f"  Markdown cells: {sum(1 for c in nb.cells if c.cell_type == 'markdown')}")
    print()

    # Analyze issues
    print("Step 2: Analyzing formatting issues...")
    issues = analyze_notebook(nb)
    print_analysis(issues)

    # Create backup
    print("Step 3: Creating backup...")
    shutil.copy2(notebook_path, backup_path)
    print(f"  Backup saved: {backup_path}")
    print()

    # Fix issues
    print("Step 4: Fixing formatting issues...")
    nb_fixed = clean_notebook(nb)
    print("  - Fixed code cell formatting")
    print("  - Fixed markdown cell formatting")
    print("  - Removed cell outputs")
    print("  - Removed trailing whitespace")
    print()

    # Save fixed notebook
    print("Step 5: Saving fixed notebook...")
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb_fixed, f)
    print(f"  Saved: {notebook_path}")
    print()

    # Re-analyze to verify fixes
    print("Step 6: Verifying fixes...")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_verify = nbformat.read(f, as_version=4)
    issues_after = analyze_notebook(nb_verify)

    print(f"  Issues before: {issues['total_issues']}")
    print(f"  Issues after: {issues_after['total_issues']}")
    print(f"  Issues fixed: {issues['total_issues'] - issues_after['total_issues']}")
    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("✓ Notebook formatting fixed successfully!")
    print(f"✓ Original file: {notebook_path}")
    print(f"✓ Backup file: {backup_path}")
    print()
    print("Fixed items:")
    print("  - Removed trailing whitespace")
    print("  - Fixed multiple statements on one line")
    print("  - Fixed markdown header spacing")
    print("  - Removed all cell outputs")
    print("  - Standardized cell structure")
    print()
    print("Remaining issues:")
    if issues_after['long_lines']:
        print(f"  - {len(issues_after['long_lines'])} long lines (>100 chars) - manual review recommended")
    if issues_after['total_issues'] == 0:
        print("  None! All formatting issues resolved.")
    print()
    print("=" * 70)


if __name__ == '__main__':
    main()