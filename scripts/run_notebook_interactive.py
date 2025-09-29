"""
Execute notebook cell by cell with error handling and progress tracking
"""
import json
import sys
import subprocess
import time
from pathlib import Path

def run_notebook_interactive(notebook_path, output_path):
    """Execute notebook cell by cell using jupyter nbconvert"""

    # Read notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    total_cells = len(nb['cells'])
    code_cells = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']

    print(f"Notebook: {notebook_path}")
    print(f"Total cells: {total_cells}")
    print(f"Code cells: {len(code_cells)}")
    print("="*70)

    # Execute using jupyter nbconvert with execute preprocessor
    print("\nExecuting notebook...")
    start_time = time.time()

    try:
        result = subprocess.run([
            'python', '-m', 'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--output', Path(output_path).name,
            '--ExecutePreprocessor.timeout=1800',  # 30 min per cell
            '--ExecutePreprocessor.kernel_name=python3',
            '--inplace' if output_path == notebook_path else '--output-dir=' + str(Path(output_path).parent),
            notebook_path
        ], capture_output=True, text=True, encoding='utf-8', errors='replace')

        if result.returncode == 0:
            elapsed = time.time() - start_time
            print(f"\nSUCCESS: Notebook executed in {elapsed:.1f}s")
            print(f"Output saved to: {output_path}")
            return True
        else:
            print(f"\nERROR: Notebook execution failed")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False

    except Exception as e:
        print(f"\nEXCEPTION: {e}")
        return False

if __name__ == '__main__':
    notebook_path = sys.argv[1] if len(sys.argv) > 1 else 'notebooks/03_injection_train.ipynb'
    output_path = sys.argv[2] if len(sys.argv) > 2 else 'notebooks/03_injection_train_executed.ipynb'

    success = run_notebook_interactive(notebook_path, output_path)
    sys.exit(0 if success else 1)