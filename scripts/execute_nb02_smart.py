import json
import sys
import os
import subprocess
import traceback
from datetime import datetime

# Set paths
NOTEBOOK_PATH = r'C:\Users\tingy\Desktop\dev\exoplanet-starter\notebooks\02_bls_baseline_LOCAL.ipynb'
OUTPUT_DIR = r'C:\Users\tingy\Desktop\dev\exoplanet-starter\outputs'
LOG_FILE = os.path.join(OUTPUT_DIR, 'nb02_execution_log.txt')

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_message(msg):
    """Log message to both console and file"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    full_msg = f'[{timestamp}] {msg}'
    print(full_msg)
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(full_msg + '\n')

def is_shell_command(code):
    """Check if code is a shell command"""
    stripped = code.strip()
    return stripped.startswith('!') or stripped.startswith('%')

def is_colab_code(code):
    """Check if code is Colab-specific"""
    return 'google.colab' in code or 'drive.mount' in code

def execute_shell_command(code):
    """Execute shell command"""
    # Remove the ! prefix
    cmd = code.strip()[1:].strip()
    log_message(f'  Executing shell: {cmd[:80]}')
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            log_message(f'  Shell command succeeded')
            return True
        else:
            log_message(f'  Shell command failed: {result.stderr[:200]}')
            return False
    except Exception as e:
        log_message(f'  Shell command error: {str(e)}')
        return False

# Load the notebook
log_message('Loading notebook...')
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract code cells
code_cells = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if source.strip():
            code_cells.append({
                'cell_index': i,
                'code': source
            })

log_message(f'Found {len(code_cells)} code cells to execute')
log_message('=' * 80)

# Execute cells
namespace = {}
successful_cells = 0
failed_cells = 0
skipped_cells = 0

for idx, cell_data in enumerate(code_cells, 1):
    cell_idx = cell_data['cell_index']
    code = cell_data['code']
    
    log_message(f'\nExecuting Code Cell {idx}/{len(code_cells)} (Notebook Cell #{cell_idx})')
    
    # Show first line of code
    first_line = code.split('\n')[0][:80]
    log_message(f'  First line: {first_line}')
    
    # Handle different code types
    if is_colab_code(code):
        log_message(f'  Status: SKIPPED (Colab-specific code)')
        skipped_cells += 1
        continue
    
    if is_shell_command(code):
        # Handle shell commands
        success = execute_shell_command(code)
        if success:
            successful_cells += 1
            log_message(f'  Status: SUCCESS')
        else:
            failed_cells += 1
            log_message(f'  Status: FAILED (shell command)')
        continue
    
    try:
        # Execute the code
        exec(code, namespace)
        successful_cells += 1
        log_message(f'  Status: SUCCESS')
        
        # Check for key variables after certain cells
        if 'df' in namespace and hasattr(namespace.get('df'), 'shape'):
            log_message(f'  Dataset shape: {namespace["df"].shape}')
        if 'features_df' in namespace and hasattr(namespace.get('features_df'), 'shape'):
            log_message(f'  Features shape: {namespace["features_df"].shape}')
        if 'results' in namespace and isinstance(namespace.get('results'), list):
            log_message(f'  Results count: {len(namespace["results"])}')
        if 'successful_count' in namespace:
            log_message(f'  Successful samples: {namespace["successful_count"]}')
        if 'failed_count' in namespace:
            log_message(f'  Failed samples: {namespace["failed_count"]}')
            
    except Exception as e:
        failed_cells += 1
        log_message(f'  Status: FAILED')
        log_message(f'  Error: {str(e)}')
        error_trace = traceback.format_exc()
        # Only log first 500 chars of traceback
        log_message(f'  Traceback:\n{error_trace[:500]}')
        
        # Continue on certain errors
        if any(x in str(e).lower() for x in ['no module', 'cannot import', 'import error']):
            log_message('  Note: Import error, but continuing...')
        elif 'file not found' in str(e).lower() or 'no such file' in str(e).lower():
            log_message('  Note: File not found, but continuing...')
        else:
            log_message('  Critical error - stopping execution')
            break

log_message('\n' + '=' * 80)
log_message('=== EXECUTION SUMMARY ===')
log_message(f'Total cells: {len(code_cells)}')
log_message(f'Successful: {successful_cells}')
log_message(f'Failed: {failed_cells}')
log_message(f'Skipped: {skipped_cells}')

# Final status
if 'df' in namespace and hasattr(namespace.get('df'), 'shape'):
    log_message(f'\nFinal dataset shape: {namespace["df"].shape}')
if 'features_df' in namespace and hasattr(namespace.get('features_df'), 'shape'):
    log_message(f'Final features shape: {namespace["features_df"].shape}')
if 'results' in namespace and isinstance(namespace.get('results'), list):
    log_message(f'Final results count: {len(namespace["results"])}')
if 'successful_count' in namespace:
    log_message(f'Total successful samples: {namespace["successful_count"]}')
if 'failed_count' in namespace:
    log_message(f'Total failed samples: {namespace["failed_count"]}')

log_message(f'\nLog file: {LOG_FILE}')
log_message('Execution complete!')
