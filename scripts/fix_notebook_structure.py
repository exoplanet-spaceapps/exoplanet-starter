"""
Fix Jupyter Notebook Structure for 02_bls_baseline_COLAB_ENHANCED.ipynb

Fixes:
1. Remove test mode cells (Cell 14-15)
2. Fix cell types (code vs markdown)
3. Remove duplicate cells
4. Renumber cells properly
"""

import json
import sys
from pathlib import Path

# Set UTF-8 encoding for console output
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def fix_notebook_structure():
    notebook_path = Path(__file__).parent.parent / "notebooks" / "02_bls_baseline_COLAB_ENHANCED.ipynb.backup"

    # Load notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    print(f"Original notebook has {len(nb['cells'])} cells")

    # Define the correct cell structure by index
    # Format: (cell_type, keep/remove, description)
    cell_plan = [
        ("markdown", True, "Title"),
        ("markdown", True, "Cell 1 title - Package Installation"),
        ("code", True, "Cell 1 code - Package installation"),
        ("markdown", True, "Cell 2 title - Environment Check"),
        ("code", True, "Cell 2 code - Environment check"),
        ("markdown", True, "Cell 3 title - Google Drive Setup"),
        ("code", True, "Cell 3 code - Google Drive setup"),
        ("markdown", True, "Cell 4 title - CheckpointManager"),
        ("code", True, "Cell 4 code - CheckpointManager class"),
        ("markdown", True, "Cell 5 title - Parallel Processing"),
        ("code", True, "Cell 5 code - Parallel processing imports"),
        ("markdown", True, "Cell 6 title - Feature Extraction"),
        ("code", True, "Cell 6 code - Feature extraction function"),
        ("code", True, "Cell 7 code - Load dataset"),
        ("markdown", False, "DELETE - Test mode title"),
        ("code", False, "DELETE - Test mode code"),
        ("code", True, "Cell 8 code - Parallel batch processing (FIX: was markdown)"),
        ("code", False, "DELETE - Duplicate batch processing"),
        ("code", True, "Cell 9 code - Execute extraction (FIX: was markdown)"),
        ("code", False, "DELETE - Duplicate execution"),
        ("markdown", True, "Cell 10 title - Progress Monitoring"),
        ("code", True, "Cell 10 code - Progress monitoring"),
        ("markdown", True, "Cell 11 title - Validate Results"),
        ("code", True, "Cell 11 code - Validate results"),
        ("markdown", True, "Cell 12 title - Cleanup"),
        ("code", True, "Cell 12 code - Cleanup"),
        ("markdown", True, "Cell 13 title - Download Results"),
        ("markdown", True, "Usage instructions"),
        ("markdown", False, "DELETE - Duplicate documentation"),
    ]

    # Process cells
    new_cells = []
    removed_count = 0
    fixed_count = 0

    for idx, cell in enumerate(nb["cells"]):
        if idx >= len(cell_plan):
            print(f"⚠️ Unexpected cell at index {idx}")
            continue

        correct_type, keep, description = cell_plan[idx]

        if not keep:
            print(f"🗑️ Removing cell {idx}: {description}")
            removed_count += 1
            continue

        # Fix cell type if needed
        if cell["cell_type"] != correct_type:
            print(f"🔧 Fixing cell {idx}: {cell['cell_type']} -> {correct_type} ({description})")
            cell["cell_type"] = correct_type
            fixed_count += 1

            # Adjust cell structure for code cells
            if correct_type == "code":
                if "execution_count" not in cell:
                    cell["execution_count"] = None
                if "outputs" not in cell:
                    cell["outputs"] = []
            else:
                # Remove code-specific fields from markdown cells
                if "execution_count" in cell:
                    del cell["execution_count"]
                if "outputs" in cell:
                    del cell["outputs"]

        new_cells.append(cell)

    # Update notebook cells
    nb["cells"] = new_cells

    # Save fixed notebook
    output_path = notebook_path.parent / "02_bls_baseline_COLAB_ENHANCED.ipynb"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n✅ Fixed notebook saved: {output_path}")
    print(f"\n📊 Summary:")
    print(f"   Original cells: {len(nb['cells']) + removed_count}")
    print(f"   Final cells: {len(new_cells)}")
    print(f"   Removed: {removed_count} cells")
    print(f"   Fixed cell types: {fixed_count} cells")
    print(f"\n✨ Notebook structure fixed!")
    print(f"\n📋 Final structure:")
    print(f"   - Cell 1: Package installation")
    print(f"   - Cell 2: Environment check")
    print(f"   - Cell 3: Google Drive setup")
    print(f"   - Cell 4: CheckpointManager")
    print(f"   - Cell 5: Parallel processing")
    print(f"   - Cell 6: Feature extraction")
    print(f"   - Cell 7: Load dataset")
    print(f"   - Cell 8: Parallel batch processing")
    print(f"   - Cell 9: Execute extraction")
    print(f"   - Cell 10: Progress monitoring")
    print(f"   - Cell 11: Validate results")
    print(f"   - Cell 12: Cleanup")
    print(f"   - Cell 13: Download results & documentation")


if __name__ == "__main__":
    fix_notebook_structure()