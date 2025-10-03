"""
Execute notebook 04 with proper path setup
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

print(f"✅ Python path configured:")
print(f"   - {project_root}")
print(f"   - {project_root / 'src'}")

# Change to notebooks directory
os.chdir(project_root / 'notebooks')

# Execute notebook
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebook_path = project_root / 'notebooks' / '04_newdata_inference.ipynb'

print(f"\n📓 Loading notebook: {notebook_path}")

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Configure executor
ep = ExecutePreprocessor(
    timeout=600,
    kernel_name='python3',
    allow_errors=False  # Stop on first error
)

print(f"\n🚀 Executing notebook...")
print("=" * 60)

try:
    # Execute
    ep.preprocess(nb, {'metadata': {'path': str(project_root / 'notebooks')}})

    # Save with outputs
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print("\n" + "=" * 60)
    print("✅ Notebook executed successfully!")

    # Count executed cells
    executed = sum(1 for cell in nb.cells if cell.cell_type == 'code' and cell.get('execution_count'))
    total = sum(1 for cell in nb.cells if cell.cell_type == 'code')

    print(f"📊 Executed: {executed}/{total} cells")
    print(f"📁 Output saved to: {notebook_path}")

except Exception as e:
    print(f"\n❌ Error during execution:")
    print(f"   {type(e).__name__}: {e}")

    # Save notebook with partial execution
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

    print(f"\n⚠️ Partial execution saved to: {notebook_path}")
    sys.exit(1)