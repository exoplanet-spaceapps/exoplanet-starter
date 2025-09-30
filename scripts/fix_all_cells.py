#!/usr/bin/env python3
"""Fix all remaining formatting issues in specific cells."""

import nbformat
from pathlib import Path
import re


def fix_cell_4():
    """Fix cell 4 - Setup paths."""
    return """# Setup paths
if IN_COLAB:
    PROJECT_ROOT = Path('/content/exoplanet-starter')
else:
    PROJECT_ROOT = Path.cwd().parent if 'notebooks' in str(Path.cwd()) else Path.cwd()

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'app'))

print(f"Project root: {PROJECT_ROOT}")"""


def fix_cell_2():
    """Fix cell 2 - Check environment."""
    return """# Check environment
try:
    import google.colab
    IN_COLAB = True
    print("Running in Google Colab")
except:
    IN_COLAB = False
    print("Running locally")"""


def expand_condensed_code(source: str) -> str:
    """Expand any condensed code statements."""
    lines = []

    for line in source.split('\n'):
        line = line.rstrip()

        # Pattern: if ...:    ...
        if_match = re.match(r'^(\s*)(if .+?):\s{2,}(.+)$', line)
        if if_match:
            indent, condition, body = if_match.groups()
            lines.append(f"{indent}{condition}:")
            lines.append(f"{indent}    {body}")
            continue

        # Pattern: else:    ...
        else_match = re.match(r'^(\s*)(else):\s{2,}(.+)$', line)
        if else_match:
            indent, keyword, body = else_match.groups()
            lines.append(f"{indent}{keyword}:")
            lines.append(f"{indent}    {body}")
            continue

        # Pattern: try:    ...
        try_match = re.match(r'^(\s*)(try):\s{2,}(.+)$', line)
        if try_match:
            indent, keyword, body = try_match.groups()
            lines.append(f"{indent}{keyword}:")
            lines.append(f"{indent}    {body}")
            continue

        # Pattern: except:    ...
        except_match = re.match(r'^(\s*)(except[^:]*?):\s{2,}(.+)$', line)
        if except_match:
            indent, keyword, body = except_match.groups()
            lines.append(f"{indent}{keyword}:")
            lines.append(f"{indent}    {body}")
            continue

        lines.append(line)

    return '\n'.join(lines)


def main():
    """Fix all problem cells."""
    nb_path = Path('notebooks/03_injection_train_FIXED.ipynb')

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    print("Fixing cells...")

    # Fix specific cells with custom fixes
    fixes_applied = []

    # Cell 2 (index 1)
    nb.cells[1].source = fix_cell_2()
    fixes_applied.append(2)

    # Cell 4 (index 3)
    nb.cells[3].source = fix_cell_4()
    fixes_applied.append(4)

    # Auto-fix all other cells
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code' and i not in [1, 3]:  # Skip manually fixed cells
            original = cell.source
            fixed = expand_condensed_code(original)
            if fixed != original:
                cell.source = fixed
                fixes_applied.append(i + 1)

    # Save
    with open(nb_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"Fixed {len(fixes_applied)} cells: {fixes_applied}")
    print("Done!")


if __name__ == '__main__':
    main()