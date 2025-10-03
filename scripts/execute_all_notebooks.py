#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Execute all notebooks (03, 04, 05) using papermill
Handles encoding issues and provides detailed progress reports
"""

import sys
import os
import json
from pathlib import Path
import subprocess
import time

# Force UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

def check_notebook_execution(notebook_path):
    """Check how many cells were executed in a notebook"""
    with open(notebook_path, encoding='utf-8') as f:
        data = json.load(f)

    code_cells = [c for c in data['cells'] if c['cell_type'] == 'code']
    executed = sum(1 for c in code_cells if c.get('execution_count') is not None)

    return executed, len(code_cells)

def execute_notebook(input_nb, output_nb, timeout=600):
    """Execute a single notebook using papermill"""
    print(f"\n{'='*70}")
    print(f"Executing: {input_nb}")
    print(f"Output: {output_nb}")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        # Use papermill with UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'

        # Execute from notebooks directory to ensure proper imports
        cwd = Path(input_nb).parent

        result = subprocess.run([
            sys.executable, '-X', 'utf8', '-m', 'papermill',
            str(input_nb.name),
            str(output_nb.name),
            '--log-output',
            f'--execution-timeout', str(timeout)
        ], env=env, capture_output=True, text=True, encoding='utf-8', cwd=str(cwd))

        elapsed = time.time() - start_time

        if result.returncode == 0:
            executed, total = check_notebook_execution(output_nb)
            print(f"\n[SUCCESS] Executed {executed}/{total} cells in {elapsed:.1f}s")
            return True, executed, total
        else:
            print(f"\n[FAILED] Execution failed after {elapsed:.1f}s")
            print("STDERR:", result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return False, 0, 0

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] Exception after {elapsed:.1f}s: {e}")
        return False, 0, 0

def main():
    """Execute all notebooks"""
    project_root = Path(__file__).parent.parent
    notebooks_dir = project_root / 'notebooks'

    # Change to project root
    os.chdir(str(project_root))
    print(f"Working directory: {os.getcwd()}")

    # Define notebooks to execute
    notebooks = [
        {
            'name': '03_injection_train',
            'input': notebooks_dir / '03_injection_train.ipynb',
            'output': notebooks_dir / '03_injection_train_FINAL.ipynb',
            'timeout': 600
        },
        {
            'name': '04_newdata_inference',
            'input': notebooks_dir / '04_newdata_inference.ipynb',
            'output': notebooks_dir / '04_newdata_inference_FINAL.ipynb',
            'timeout': 300
        },
        {
            'name': '05_metrics_dashboard',
            'input': notebooks_dir / '05_metrics_dashboard.ipynb',
            'output': notebooks_dir / '05_metrics_dashboard_FINAL.ipynb',
            'timeout': 300
        }
    ]

    # Execute each notebook
    results = []
    for nb in notebooks:
        success, executed, total = execute_notebook(
            nb['input'],
            nb['output'],
            nb['timeout']
        )

        results.append({
            'name': nb['name'],
            'success': success,
            'executed': executed,
            'total': total,
            'percentage': (executed / total * 100) if total > 0 else 0
        })

    # Print summary
    print("\n" + "="*70)
    print("EXECUTION SUMMARY")
    print("="*70)

    all_success = True
    for r in results:
        status = "SUCCESS" if r['success'] else "FAILED"
        print(f"\n{r['name']}: {status}")
        print(f"  Executed: {r['executed']}/{r['total']} cells ({r['percentage']:.1f}%)")

        if not r['success']:
            all_success = False

    # Replace original files if all succeeded
    if all_success:
        print("\n" + "="*70)
        print("All notebooks executed successfully!")
        print("Replacing original files...")
        print("="*70)

        for nb in notebooks:
            if nb['output'].exists():
                print(f"Replacing {nb['input'].name}...")
                nb['output'].replace(nb['input'])
                print(f"  [OK] {nb['input'].name} updated with execution results")

        print("\n[SUCCESS] All notebooks have been updated with execution results!")
        print("Ready to commit and push.")
        return 0
    else:
        print("\n" + "="*70)
        print("Some notebooks failed to execute.")
        print("Original files were NOT replaced.")
        print("="*70)
        return 1

if __name__ == '__main__':
    sys.exit(main())