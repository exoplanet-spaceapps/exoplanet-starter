import os
import time
import glob
from datetime import datetime

# Paths
base_dir = r'C:\Users\tingy\Desktop\dev\exoplanet-starter'
outputs_dir = os.path.join(base_dir, 'outputs')
data_dir = os.path.join(base_dir, 'data')
log_file = os.path.join(outputs_dir, 'nb02_full_run.log')

print(f'=== Notebook 02 Progress Monitor ===')
print(f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
print(f'\n--- Log File ---')
if os.path.exists(log_file):
    size = os.path.getsize(log_file)
    print(f'Log file: {log_file}')
    print(f'Size: {size} bytes')
    if size > 0:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            print(f'Lines: {len(lines)}')
            if lines:
                print('\nLast 10 lines:')
                for line in lines[-10:]:
                    print(f'  {line.rstrip()}')
    else:
        print('Log file is empty (output may be buffered)')
else:
    print('Log file not yet created')

print(f'\n--- Output Files ---')
# Check for any output files
output_files = glob.glob(os.path.join(outputs_dir, '*.csv'))
output_files += glob.glob(os.path.join(outputs_dir, '*.pkl'))
output_files += glob.glob(os.path.join(outputs_dir, '*.npy'))
output_files += glob.glob(os.path.join(data_dir, '*features*.csv'))
output_files += glob.glob(os.path.join(data_dir, '*bls*.csv'))

if output_files:
    print(f'Found {len(output_files)} output files:')
    for f in sorted(output_files, key=os.path.getmtime, reverse=True)[:10]:
        mtime = datetime.fromtimestamp(os.path.getmtime(f))
        size = os.path.getsize(f)
        print(f'  {os.path.basename(f):40s} {size:>12,} bytes  {mtime.strftime("%H:%M:%S")}')
else:
    print('No output files found yet')

print(f'\n--- Data Directory ---')
data_files = glob.glob(os.path.join(data_dir, '*.csv'))
print(f'CSV files in data/: {len(data_files)}')
for f in sorted(data_files, key=os.path.getmtime, reverse=True)[:5]:
    mtime = datetime.fromtimestamp(os.path.getmtime(f))
    size = os.path.getsize(f)
    print(f'  {os.path.basename(f):40s} {size:>12,} bytes  {mtime.strftime("%Y-%m-%d %H:%M")}')

print(f'\n--- Process Check ---')
# Try to detect if processing is happening by checking temp files or cache
cache_dirs = [
    os.path.join(os.path.expanduser('~'), '.lightkurve', 'cache'),
    os.path.join(base_dir, '.cache'),
    os.path.join(base_dir, 'temp')
]

for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        files = glob.glob(os.path.join(cache_dir, '*'))
        if files:
            print(f'Cache: {cache_dir} - {len(files)} files')
            # Check most recent
            recent = max(files, key=os.path.getmtime)
            mtime = datetime.fromtimestamp(os.path.getmtime(recent))
            print(f'  Most recent: {os.path.basename(recent)} at {mtime.strftime("%H:%M:%S")}')
