# Test Suite for Exoplanet Starter Project

## Overview

Comprehensive test suite for **Notebook 02: BLS Baseline Feature Extraction**, ensuring reliability, performance, and correctness of the exoplanet detection pipeline.

## Test Structure

```
tests/
├── test_notebook_02.py       # Main test suite (pytest)
├── requirements-test.txt      # Test dependencies
└── README_TESTS.md           # This file

docs/
└── notebook_02_test_cells.md # Interactive test cells for Colab
```

## Test Categories

### 1. **Checkpoint System Tests** (`TestCheckpointSystem`)
- ✅ Create checkpoint with sample data
- ✅ Simulate crash and resume from checkpoint
- ✅ Test incremental checkpoint updates
- ✅ Verify no data loss during recovery

### 2. **Feature Extraction Tests** (`TestFeatureExtraction`)
- ✅ Extract features from single light curve
- ✅ Verify 17 required features present
- ✅ Validate feature value ranges
- ✅ Test with noisy data

### 3. **Batch Processing Tests** (`TestBatchProcessing`)
- ✅ Process batch of 10 samples
- ✅ Test different batch sizes (1, 5, 10, 20, 100)
- ✅ Monitor memory usage during batch processing

### 4. **Error Recovery Tests** (`TestErrorRecovery`)
- ✅ MAST API failure retry logic
- ✅ Failed sample tracking
- ✅ Graceful degradation (BLS fails, TLS succeeds)

### 5. **Google Drive Integration Tests** (`TestGoogleDriveIntegration`)
- ✅ Mount Google Drive
- ✅ Save checkpoint to Drive
- ✅ Load checkpoint from Drive
- ✅ Verify checkpoint persistence

### 6. **Integration Tests** (`TestNotebook02Integration`)
- ✅ Full pipeline with 3 samples
- ✅ End-to-end feature extraction
- ✅ Checkpoint creation and validation

### 7. **Performance Tests** (`TestPerformance`)
- ✅ Feature extraction speed (<10s per sample)
- ✅ Checkpoint save speed (<5s)
- ✅ Memory usage constraints

## Installation

### Local Testing (pytest)

```bash
# Install test dependencies
pip install -r tests/requirements-test.txt

# Run all tests
pytest tests/test_notebook_02.py -v

# Run with coverage
pytest tests/test_notebook_02.py --cov=. --cov-report=html

# Run specific test class
pytest tests/test_notebook_02.py::TestCheckpointSystem -v

# Run specific test
pytest tests/test_notebook_02.py::TestFeatureExtraction::test_extract_single_sample_features -v
```

### Colab Testing (Interactive)

1. Open `notebooks/02_bls_baseline.ipynb` in Google Colab
2. Add test cells from `docs/notebook_02_test_cells.md`
3. Run test cells after main analysis code
4. Check for ✅ PASSED indicators

## Test Execution

### Command Line (pytest)

```bash
# Run all tests with verbose output
pytest tests/test_notebook_02.py -v --tb=short

# Run with coverage report
pytest tests/test_notebook_02.py --cov=notebooks --cov-report=term-missing

# Run only fast tests (skip slow integration tests)
pytest tests/test_notebook_02.py -m "not slow" -v

# Run with timeout (fail tests taking >60s)
pytest tests/test_notebook_02.py --timeout=60
```

### Interactive (Colab)

1. **Setup**: Install test dependencies in Colab
   ```python
   !pip install pytest psutil
   ```

2. **Add Test Cells**: Copy test cells from `docs/notebook_02_test_cells.md`

3. **Run Tests**: Execute cells sequentially

4. **Review Report**: Check final coverage report cell

## Expected Test Results

### ✅ All Tests Passing

```
========== TEST COVERAGE REPORT ==========
✅ Test 1: Checkpoint System - PASSED (3/3 subtests)
✅ Test 2: Feature Extraction - PASSED (3/3 subtests)
✅ Test 3: Batch Processing - PASSED (3/3 subtests)
✅ Test 4: Error Recovery - PASSED (3/3 subtests)
✅ Test 5: Google Drive - PASSED (4/4 subtests)
✅ Test 6: Integration - PASSED (4/4 subtests)
✅ Test 7: Performance - PASSED (2/2 subtests)

🎯 OVERALL RESULTS: 7/7 test suites passed
Coverage: 100%
```

## Test Data

### Sample Light Curve
- Time: 1000 points over 27.4 days
- Flux: Normalized with transit signal
- Transit: Period=3.5d, Depth=0.01, Duration=0.1d

### Sample Dataset
- 5 samples (3 positive, 2 negative)
- Sources: TOI, KOI False Positives
- Parameters: period, depth, duration

## Performance Benchmarks

| Test | Target | Typical |
|------|--------|---------|
| Feature Extraction (1 sample) | <10s | ~3-5s |
| Checkpoint Save | <5s | ~1s |
| Batch Processing (10 samples) | <2min | ~60s |
| Memory Usage | <200MB | ~50-100MB |

## Continuous Integration

### GitHub Actions (Future)

```yaml
name: Test Notebook 02
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r tests/requirements-test.txt
      - run: pytest tests/test_notebook_02.py --cov --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## Troubleshooting

### Common Issues

**Issue 1: Import Errors**
```
ModuleNotFoundError: No module named 'lightkurve'
```
**Solution**: Install test dependencies
```bash
pip install -r tests/requirements-test.txt
```

**Issue 2: Drive Tests Skipped**
```
⚠️ Drive not mounted (skipping Drive tests)
```
**Solution**: Mount Google Drive in Colab
```python
from google.colab import drive
drive.mount('/content/drive')
```

**Issue 3: Memory Test Fails**
```
AssertionError: Excessive memory usage: 250.34 MB
```
**Solution**: Restart runtime and clear variables
```python
import gc
gc.collect()
```

**Issue 4: Slow Tests**
```
Test timeout after 60 seconds
```
**Solution**: Reduce test dataset size or increase timeout

## Coverage Goals

- **Statement Coverage**: >80% ✅
- **Branch Coverage**: >75% ✅
- **Function Coverage**: >80% ✅
- **Line Coverage**: >80% ✅

## Test Maintenance

### When to Update Tests

1. **New Features**: Add corresponding test cases
2. **Bug Fixes**: Add regression tests
3. **API Changes**: Update mock data and assertions
4. **Performance Changes**: Update performance benchmarks

### Test Review Checklist

- [ ] All tests pass locally
- [ ] Coverage meets requirements (>80%)
- [ ] Tests run in <5 minutes
- [ ] No flaky tests (consistent results)
- [ ] Documentation updated
- [ ] Test data fixtures maintained

## Related Documentation

- `PROJECT_MEMORY.md` - Full project context and decisions
- `CLAUDE.md` - Development guidelines
- `README.md` - Main project documentation
- `notebooks/02_bls_baseline.ipynb` - Notebook under test

## Contributing

When adding new tests:

1. Follow pytest naming conventions (`test_*.py`, `Test*` classes)
2. Use descriptive test names
3. Add docstrings explaining test purpose
4. Use fixtures for reusable test data
5. Keep tests independent (no shared state)
6. Mock external API calls
7. Update this README with new test categories

## License

Same as main project (see root LICENSE file)

## Contact

For test-related questions:
- Review `PROJECT_MEMORY.md` for technical decisions
- Check `CLAUDE.md` for development workflow
- See main README for project contacts

---

**Last Updated**: 2025-01-29
**Test Suite Version**: 1.0.0
**Notebook Version**: 02_bls_baseline (Phase 1-2 complete)