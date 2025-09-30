import json
import sys
import os
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

for idx, cell_data in enumerate(code_cells, 1):
    cell_idx = cell_data['cell_index']
    code = cell_data['code']
    
    log_message(f'\nExecuting Code Cell {idx}/{len(code_cells)} (Notebook Cell #{cell_idx})')
    
    # Show first line of code
    first_line = code.split('\n')[0][:80]
    log_message(f'  First line: {first_line}')
    
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
            
    except Exception as e:
        failed_cells += 1
        log_message(f'  Status: FAILED')
        log_message(f'  Error: {str(e)}')
        log_message(f'  Traceback:\n{traceback.format_exc()}')
        
        # For some errors, we might want to continue
        if 'google.colab' in str(e) or 'drive.mount' in str(e):
            log_message('  Note: Colab-related error, continuing...')
        else:
            log_message('  Critical error - stopping execution')
            break

log_message('\n' + '=' * 80)
log_message('=== EXECUTION SUMMARY ===')
log_message(f'Total cells: {len(code_cells)}')
log_message(f'Successful: {successful_cells}')
log_message(f'Failed: {failed_cells}')

# Final status
if 'df' in namespace:
    log_message(f'\nFinal dataset shape: {namespace["df"].shape}')
if 'features_df' in namespace:
    log_message(f'Final features shape: {namespace["features_df"].shape}')
if 'results' in namespace:
    log_message(f'Final results count: {len(namespace["results"])}')

log_message(f'\nLog file: {LOG_FILE}')
log_message('Execution complete!')
