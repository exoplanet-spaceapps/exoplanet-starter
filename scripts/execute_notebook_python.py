"""
Execute notebook programmatically using nbformat and ExecutePreprocessor
"""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import sys
from pathlib import Path
import time

def execute_notebook(input_path, output_path, timeout=1800):
    """Execute a Jupyter notebook"""

    print(f"Reading notebook: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    total_cells = len(nb.cells)
    code_cells = sum(1 for c in nb.cells if c.cell_type == 'code')

    print(f"Total cells: {total_cells}")
    print(f"Code cells: {code_cells}")
    print("="*70)

    # Create executor
    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')

    print("\nExecuting notebook...")
    start_time = time.time()

    try:
        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': str(Path(input_path).parent)}})

        elapsed = time.time() - start_time
        print(f"\nSUCCESS: All cells executed in {elapsed:.1f}s")

        # Save the executed notebook
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        print(f"Output saved to: {output_path}")
        return True

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\nERROR after {elapsed:.1f}s: {str(e)}")

        # Save partial results
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        print(f"Partial results saved to: {output_path}")

        # Count executed cells
        executed = sum(1 for c in nb.cells if c.cell_type == 'code' and c.get('outputs'))
        print(f"Cells executed: {executed}/{code_cells}")

        return False

if __name__ == '__main__':
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'notebooks/03_injection_train.ipynb'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'notebooks/03_injection_train_executed.ipynb'

    success = execute_notebook(input_path, output_path)
    sys.exit(0 if success else 1)