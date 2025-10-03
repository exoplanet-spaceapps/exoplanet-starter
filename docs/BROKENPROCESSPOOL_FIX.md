# BrokenProcessPool Error Fix

## Problem

`BrokenProcessPool: A process in the process pool was terminated abruptly while the future was running or pending.`

This error occurs in parallel processing when:
1. A worker process crashes due to memory issues
2. External API calls (MAST) timeout or fail
3. Too many concurrent connections overwhelm the system
4. Light curve downloads fail with uncaught exceptions

## Root Causes

### 1. MAST API Instability
- Cached files with size 0 (corrupted downloads)
- Connection timeouts during `search_lightcurve()`
- Too many concurrent MAST requests

### 2. Memory Issues
- Each worker loads large light curve data
- NumPy arrays accumulate in memory
- Garbage collection not frequent enough

### 3. Uncaught Exceptions in Workers
- Exceptions in worker processes kill the entire pool
- No proper cleanup on worker failure
- Missing error isolation

## Solution Strategy

### Fix 1: Robust Worker Function with Timeouts
```python
from concurrent.futures import TimeoutError
import signal

def extract_single_sample_robust(args):
    """Worker with built-in timeout and error isolation"""
    idx, row, run_bls, run_tls = args

    try:
        # Set timeout for MAST operations
        with timeout(seconds=120):
            # ... extraction logic
            pass
    except TimeoutError:
        return (int(idx), None, "Timeout: MAST download took >120s")
    except Exception as e:
        return (int(idx), None, f"Worker error: {str(e)}")
```

### Fix 2: Reduce Concurrent MAST Requests
```python
# Instead of 12 workers, use fewer for MAST-heavy operations
n_workers = min(4, mp.cpu_count())  # Limit to 4 concurrent MAST requests
```

### Fix 3: Add Retry Mechanism
```python
max_retries = 3
for attempt in range(max_retries):
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # ... process batch
            break
    except BrokenProcessPool:
        if attempt < max_retries - 1:
            print(f"⚠️ Pool crashed, retrying ({attempt+1}/{max_retries})...")
            time.sleep(5)
        else:
            raise
```

### Fix 4: Clear MAST Cache
```python
import shutil

# Clear corrupted cache files
cache_dir = Path.home() / '.lightkurve' / 'cache'
if cache_dir.exists():
    print("🧹 Clearing corrupted MAST cache...")
    shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
```

### Fix 5: Sequential Fallback Mode
```python
def extract_features_batch_safe(samples_df, checkpoint_mgr, n_workers=12, ...):
    """Batch processing with automatic fallback to sequential"""

    try:
        # Try parallel processing
        return extract_features_batch_parallel(...)
    except BrokenProcessPool:
        print("⚠️ Parallel processing failed, falling back to sequential...")
        return extract_features_batch_sequential(...)
```

## Implementation Plan

1. **Immediate Fix**: Reduce `n_workers` from 12 to 4
2. **Add Timeout**: Wrap MAST calls with 120s timeout
3. **Cache Clearing**: Add cache cleanup at start
4. **Retry Logic**: Implement 3-retry mechanism
5. **Fallback Mode**: Add sequential processing as backup

## Testing

```python
# Test with reduced workers
n_workers = 4  # Instead of 12

# Test with timeout
extract_single_sample_timeout(args, timeout_seconds=120)

# Test retry mechanism
for attempt in range(3):
    try:
        result = process_batch()
        break
    except BrokenProcessPool:
        continue
```

## Expected Results

- ✅ Reduced worker pool crashes
- ✅ Better MAST API stability
- ✅ Automatic recovery on failure
- ✅ Slower but more reliable processing (4 workers = ~4x speedup)

## Alternative: Chunk-Based Processing

Instead of parallel workers, use smaller batches:

```python
# Process in smaller chunks
chunk_size = 10  # Process 10 at a time instead of 100
for chunk_start in range(0, len(batch), chunk_size):
    chunk = batch[chunk_start:chunk_start + chunk_size]
    # Process chunk sequentially or with 2-4 workers
```

## Recommendation

**For Colab Free Tier**:
- Use `n_workers=4` (not 12)
- Add 120s timeout per sample
- Clear MAST cache before starting
- Implement retry logic (3 attempts)
- Add sequential fallback

**For Colab Pro**:
- Can try `n_workers=8`
- Same timeout and retry logic
- Better memory handling

## Status

- [x] Diagnosed BrokenProcessPool cause
- [ ] Implemented robust worker function
- [ ] Added timeout mechanism
- [ ] Implemented retry logic
- [ ] Added cache clearing
- [ ] Tested with reduced workers
- [ ] Updated Notebook 02
- [ ] Committed fix