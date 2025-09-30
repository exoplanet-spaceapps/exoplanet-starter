# 📋 Google Colab Notebook Specifications (SPARC)

**Document Version**: 1.0
**Created**: 2025-09-30
**Project**: NASA Exoplanet Detection Pipeline
**Environment**: Google Colab with GPU (T4/A100)

---

## Executive Summary

This document provides comprehensive specifications for three critical Google Colab notebooks in the exoplanet detection pipeline. These specifications follow the **SPARC (Specification, Pseudocode, Architecture, Refinement, Completion)** methodology to ensure clear requirements, testable acceptance criteria, and successful implementation.

### Current Status
- **Dataset**: 11,979 samples from NASA TOI/KOI datasets
- **Issue**: Only 30 samples processed in prototype
- **Goal**: Process full dataset with production-quality notebooks

---

## 1. Notebook 02: BLS/TLS Baseline Feature Extraction

### 1.1 Specification

#### Purpose
Extract astronomical features from light curves using Box Least Squares (BLS) and Transit Least Squares (TLS) algorithms for all 11,979 samples in the supervised dataset.

#### Functional Requirements

**FR-2.1: Data Input**
- **ID**: FR-2.1
- **Description**: Load and validate supervised dataset
- **Priority**: High
- **Acceptance Criteria**:
  - Read `data/supervised_dataset.csv` (11,979 rows)
  - Validate required columns: `tid`, `toi`, `toipfx`, `tfopwg_disp`, `pl_orbper`, `pl_trandep`, `pl_trandurh`
  - Handle missing values with appropriate warnings
  - Memory footprint < 2GB for dataset loading

**FR-2.2: Light Curve Download**
- **ID**: FR-2.2
- **Description**: Retrieve TESS/Kepler light curves from MAST
- **Priority**: High
- **Acceptance Criteria**:
  - Use Lightkurve API for TESS/Kepler data
  - Support both TIC and KIC identifiers
  - Handle mission/author selection (TESS: SPOC, Kepler: Kepler)
  - Implement retry logic (3 attempts with exponential backoff)
  - Fallback to alternative sectors/quarters if primary fails
  - Download success rate > 90%

**FR-2.3: Light Curve Preprocessing**
- **ID**: FR-2.3
- **Description**: Clean and normalize light curves
- **Priority**: High
- **Acceptance Criteria**:
  - Remove NaN/Inf values
  - Flatten light curves (window_length=401 points)
  - Normalize to relative flux (mean=1)
  - Clip outliers at 5-sigma level
  - Minimum required points: 100 after cleaning

**FR-2.4: BLS Period Search**
- **ID**: FR-2.4
- **Description**: Execute Box Least Squares algorithm
- **Priority**: High
- **Acceptance Criteria**:
  - Period search range: 0.5 - 20.0 days
  - Minimum number of transits: 2
  - Duration grid: 0.01 - 0.3 fraction of period
  - Extract: period, power, depth, duration, epoch
  - Execution time: < 10 seconds per light curve

**FR-2.5: TLS Period Search**
- **ID**: FR-2.5
- **Description**: Execute Transit Least Squares algorithm
- **Priority**: High
- **Acceptance Criteria**:
  - Period search range: 0.5 - 20.0 days
  - Use BLS period as initial guess
  - Extract: period, power, depth, duration, SNR, SDE
  - Include transit shape parameters
  - Execution time: < 30 seconds per light curve

**FR-2.6: Feature Extraction**
- **ID**: FR-2.6
- **Description**: Calculate comprehensive feature set
- **Priority**: High
- **Acceptance Criteria**:
  - **Basic Statistics** (4 features):
    - `flux_mean`: Mean normalized flux
    - `flux_std`: Standard deviation of flux
    - `flux_median`: Median flux
    - `flux_mad`: Median absolute deviation
  - **Input Parameters** (4 features):
    - `input_period`: Period from TOI/KOI catalog
    - `input_duration`: Duration from catalog
    - `input_depth`: Depth from catalog
    - `input_epoch`: Epoch from catalog
  - **BLS Features** (5 features):
    - `bls_power`: BLS power statistic
    - `bls_period`: Best period from BLS
    - `bls_duration`: Transit duration (days)
    - `bls_depth`: Transit depth (fractional)
    - `bls_snr`: Signal-to-noise ratio
  - **TLS Features** (6 features):
    - `tls_power`: TLS power statistic
    - `tls_period`: Best period from TLS
    - `tls_duration`: Transit duration (hours)
    - `tls_depth`: Transit depth (ppm)
    - `tls_snr`: Signal-to-noise ratio
    - `tls_sde`: Signal detection efficiency
  - **Advanced Features** (8 features):
    - `odd_even_mismatch`: Depth difference between odd/even transits
    - `secondary_power_ratio`: Ratio of secondary to primary peak
    - `harmonic_delta_chisq`: Chi-square improvement over harmonics
    - `periodicity_strength`: Phase-folded coherence
    - `transit_symmetry`: Ingress/egress symmetry measure
    - `odd_even_depth_diff`: Absolute depth difference (ppm)
    - `phase_coverage`: Fraction of phase covered
    - `ingress_egress_asymmetry`: Shape asymmetry metric
  - **Total**: 27 features per sample

**FR-2.7: Batch Processing**
- **ID**: FR-2.7
- **Description**: Process full dataset with memory management
- **Priority**: Critical
- **Acceptance Criteria**:
  - Batch size: 100 samples per batch
  - Progress saving: Checkpoint every 100 samples
  - Resume capability: Load from last checkpoint on restart
  - Error recovery: Continue processing after individual failures
  - Memory management: Clear cache between batches
  - Progress reporting: Update every 50 samples

**FR-2.8: Output Generation**
- **ID**: FR-2.8
- **Description**: Save extracted features to CSV
- **Priority**: High
- **Acceptance Criteria**:
  - Output file: `data/bls_tls_features.csv`
  - Include all 27 feature columns
  - Include metadata: `tid`, `toi`, `label`, `success_flag`
  - CSV format with header row
  - UTF-8 encoding
  - File size estimate: ~5-10 MB

#### Non-Functional Requirements

**NFR-2.1: Performance**
- **ID**: NFR-2.1
- **Category**: Performance
- **Description**: Complete processing within acceptable timeframe
- **Measurement**:
  - Average processing time: 40 seconds per sample
  - Total estimated time: 20-30 hours for 11,979 samples
  - Memory usage: < 8GB peak
  - CPU utilization: > 80% during processing
- **Constraint**: Use Google Colab background execution

**NFR-2.2: Reliability**
- **ID**: NFR-2.2
- **Category**: Reliability
- **Description**: Handle errors gracefully without data loss
- **Measurement**:
  - Checkpoint success rate: 100%
  - Data recovery: Resume from any checkpoint
  - Error logging: All failures recorded
  - Overall success rate: > 90% of samples

**NFR-2.3: Compatibility**
- **ID**: NFR-2.3
- **Category**: Compatibility
- **Description**: Work in Google Colab environment
- **Validation**:
  - NumPy version: 1.26.4 (not 2.0+)
  - SciPy version: < 1.13
  - Lightkurve: Latest compatible version
  - TransitLeastSquares: 1.0.31+
  - Google Colab runtime: Standard or GPU

#### Dependencies

```yaml
dependencies:
  required:
    - numpy==1.26.4
    - scipy<1.13
    - pandas>=1.5.0
    - astropy>=5.0
    - lightkurve>=2.4.0
    - transitleastsquares>=1.0.31
    - matplotlib>=3.5.0
    - tqdm>=4.65.0

  optional:
    - wotan>=1.10  # Alternative detrending
    - astroquery>=0.4.6  # Direct MAST queries
```

#### Use Cases

**UC-2.1: Fresh Start Processing**
```gherkin
Feature: Process full dataset from scratch

Scenario: Start new feature extraction
  Given supervised_dataset.csv exists with 11,979 rows
  And no checkpoint files exist
  When user runs all cells in notebook
  Then system loads dataset
  And processes samples in batches of 100
  And saves checkpoint every 100 samples
  And generates bls_tls_features.csv with 11,979 rows
  And processing completes in 20-30 hours
```

**UC-2.2: Resume After Interruption**
```gherkin
Feature: Resume processing from checkpoint

Scenario: Continue after Colab disconnect
  Given checkpoint file exists at sample 5000
  And 5000 samples already processed
  When user restarts runtime and runs notebook
  Then system detects existing checkpoint
  And loads previous progress
  And resumes processing from sample 5001
  And completes remaining 6,979 samples
```

**UC-2.3: Handle Download Failures**
```gherkin
Feature: Graceful error handling

Scenario: MAST download fails for some targets
  Given processing sample at index 1234
  When Lightkurve download fails for TIC 12345678
  And retry attempts (3) all fail
  Then system logs error with TIC ID
  And marks sample as failed in checkpoint
  And continues processing next sample
  And final CSV includes success_flag column
```

#### Acceptance Criteria Summary

**✓ Definition of Done:**
- [ ] Notebook executes without errors in fresh Colab runtime
- [ ] All 11,979 samples processed (>90% success rate)
- [ ] Output file `bls_tls_features.csv` contains 27 feature columns
- [ ] Checkpoint system successfully saves/resumes progress
- [ ] Processing time < 30 hours (with background execution)
- [ ] Memory usage never exceeds 8GB
- [ ] Error handling logs all failures with TIC IDs
- [ ] Documentation includes runtime restart instructions
- [ ] GitHub push function successfully uploads results

### 1.2 Architecture

#### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Notebook 02: BLS Baseline                   │
└─────────────────────────────────────────────────────────────────┘

INPUT:
  supervised_dataset.csv (11,979 samples)
    ├── tid (TIC/KIC identifier)
    ├── label (1=planet, 0=false positive)
    └── catalog_params (period, depth, duration)

PROCESSING PIPELINE:

  ┌───────────────────┐
  │ 1. Initialization │
  │  - Mount Drive    │
  │  - Load packages  │
  │  - Check env      │
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────┐
  │ 2. Load Dataset   │
  │  - Read CSV       │
  │  - Validate cols  │
  │  - Load checkpoint│
  └────────┬──────────┘
           │
           ▼
  ┌───────────────────────────────────────┐
  │ 3. Batch Processing Loop              │
  │   (Iterate: batches of 100 samples)   │
  │                                       │
  │   For each sample:                    │
  │   ┌─────────────────────────────┐   │
  │   │ a. Download Light Curve     │   │
  │   │    - Query MAST              │   │
  │   │    - Handle sectors          │   │
  │   │    - Retry logic (3x)        │   │
  │   └──────────┬──────────────────┘   │
  │              ▼                        │
  │   ┌─────────────────────────────┐   │
  │   │ b. Preprocess               │   │
  │   │    - Remove NaNs             │   │
  │   │    - Flatten (window=401)    │   │
  │   │    - Normalize flux          │   │
  │   └──────────┬──────────────────┘   │
  │              ▼                        │
  │   ┌─────────────────────────────┐   │
  │   │ c. BLS Analysis             │   │
  │   │    - Period search           │   │
  │   │    - Extract best period     │   │
  │   │    - Calculate power/SNR     │   │
  │   └──────────┬──────────────────┘   │
  │              ▼                        │
  │   ┌─────────────────────────────┐   │
  │   │ d. TLS Analysis             │   │
  │   │    - Refined period search   │   │
  │   │    - Transit model fit       │   │
  │   │    - Extract shape params    │   │
  │   └──────────┬──────────────────┘   │
  │              ▼                        │
  │   ┌─────────────────────────────┐   │
  │   │ e. Feature Extraction       │   │
  │   │    - 27 features             │   │
  │   │    - Advanced metrics        │   │
  │   │    - Quality flags           │   │
  │   └──────────┬──────────────────┘   │
  │              │                        │
  │   ┌──────────▼──────────────────┐   │
  │   │ f. Checkpoint Save          │   │
  │   │    (every 100 samples)       │   │
  │   └─────────────────────────────┘   │
  └───────────────────────────────────────┘
           │
           ▼
  ┌───────────────────┐
  │ 4. Finalization   │
  │  - Combine batches│
  │  - Save CSV       │
  │  - Generate report│
  └────────┬──────────┘
           │
           ▼
OUTPUT:
  data/bls_tls_features.csv (11,979 rows × 31 columns)
    ├── tid, toi, label
    ├── 27 feature columns
    └── success_flag, error_message

CHECKPOINTS:
  checkpoints/bls_checkpoint_XXXX.pkl
    ├── processed_indices
    ├── features_so_far
    └── timestamp
```

#### Module Structure

```python
# Cell 1: Environment Setup
"""
- Install packages (NumPy 1.26.4, etc.)
- Instructions for runtime restart
- Environment variable setup
"""

# Cell 2: Imports and Configuration
"""
- Import all required libraries
- Define configuration parameters
- Setup logging
"""

# Cell 3: Google Drive Mount (Optional)
"""
- Mount Drive for persistent storage
- Setup checkpoint directory
"""

# Cell 4: Helper Functions
"""
def download_light_curve(tid, mission='TESS', retry=3):
    '''Download light curve with retry logic'''

def preprocess_light_curve(lc):
    '''Clean and normalize light curve'''

def run_bls(time, flux, period_range=(0.5, 20)):
    '''Execute BLS algorithm'''

def run_tls(time, flux, bls_period):
    '''Execute TLS algorithm'''

def extract_features(lc, bls_result, tls_result):
    '''Extract 27 features from results'''

def save_checkpoint(features_df, index):
    '''Save progress checkpoint'''

def load_checkpoint():
    '''Load previous checkpoint if exists'''
"""

# Cell 5: Main Processing Loop
"""
# Load dataset
df = pd.read_csv('data/supervised_dataset.csv')

# Initialize or load checkpoint
checkpoint = load_checkpoint()
start_idx = checkpoint['last_index'] if checkpoint else 0

# Batch processing
for batch_start in range(start_idx, len(df), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(df))
    batch_results = []

    for idx in range(batch_start, batch_end):
        try:
            # Process single sample
            result = process_sample(df.iloc[idx])
            batch_results.append(result)
        except Exception as e:
            # Log error and continue
            log_error(idx, e)
            continue

    # Save checkpoint
    save_checkpoint(batch_results, batch_end)
"""

# Cell 6: Results Aggregation
"""
# Combine all checkpoints
all_features = combine_checkpoints()

# Save final CSV
all_features.to_csv('data/bls_tls_features.csv', index=False)

# Generate summary report
print_summary_report(all_features)
"""

# Cell 7: Visualization (Optional)
"""
# Plot feature distributions
# Show processing statistics
# Display error summary
"""

# Cell 8: GitHub Push
"""
# Push results to GitHub
# ultimate_push_to_github_02()
"""
```

#### Checkpoint System Design

```python
# Checkpoint Format
checkpoint_schema = {
    'version': '1.0',
    'timestamp': '2025-09-30T12:34:56',
    'last_index': 5000,
    'total_processed': 5000,
    'total_failed': 234,
    'features_dataframe': pd.DataFrame(...),  # Serialized
    'config': {
        'batch_size': 100,
        'period_range': [0.5, 20.0],
        'numpy_version': '1.26.4'
    },
    'failed_samples': [
        {'index': 123, 'tid': 'TIC 12345', 'error': 'Download failed'},
        ...
    ]
}

# Checkpoint Strategy
- Save location: `/content/drive/MyDrive/exoplanet-checkpoints/`
- Filename pattern: `bls_checkpoint_{batch_start:05d}.pkl`
- Save frequency: Every 100 samples
- Retention: Keep last 3 checkpoints
- Resume logic: Detect and load most recent checkpoint
```

#### Error Handling Strategy

```python
# Error Categories and Responses

class ProcessingError:
    """Base error class"""

class DownloadError(ProcessingError):
    """Light curve download failed"""
    # Response: Retry 3 times, then skip sample

class PreprocessError(ProcessingError):
    """Preprocessing failed (too few points, etc.)"""
    # Response: Log error, mark as failed, continue

class BLSError(ProcessingError):
    """BLS algorithm failed"""
    # Response: Use default/NaN values, continue to TLS

class TLSError(ProcessingError):
    """TLS algorithm failed"""
    # Response: Use BLS values only, mark TLS as failed

class FeatureError(ProcessingError):
    """Feature extraction failed"""
    # Response: Use partial features, log warning

# Error Logging
error_log_schema = {
    'timestamp': datetime,
    'sample_index': int,
    'tid': str,
    'error_type': str,
    'error_message': str,
    'traceback': str,
    'retry_count': int
}
```

### 1.3 Implementation Pseudocode

```python
"""
Notebook 02: BLS/TLS Feature Extraction - Implementation Pseudocode
"""

# ============================================================================
# CELL 1: Package Installation
# ============================================================================
PRINT "📦 Installing required packages..."
EXECUTE pip install numpy==1.26.4 scipy'<1.13' pandas astropy
EXECUTE pip install lightkurve transitleastsquares matplotlib tqdm

PRINT "⚠️ IMPORTANT: Runtime restart required!"
PRINT "   1. Click 'Runtime' → 'Restart runtime'"
PRINT "   2. Then continue from Cell 2"
EXIT_CELL


# ============================================================================
# CELL 2: Imports and Configuration
# ============================================================================
IMPORT numpy, pandas, matplotlib, astropy, lightkurve, transitleastsquares
IMPORT json, pickle, time, datetime
FROM pathlib IMPORT Path
FROM tqdm IMPORT tqdm

# Configuration
CONFIG = {
    'BATCH_SIZE': 100,
    'CHECKPOINT_FREQ': 100,
    'PERIOD_MIN': 0.5,
    'PERIOD_MAX': 20.0,
    'DURATION_MIN': 0.01,
    'DURATION_MAX': 0.3,
    'MIN_TRANSITS': 2,
    'FLATTEN_WINDOW': 401,
    'MIN_POINTS': 100,
    'MAX_RETRIES': 3,
    'TIMEOUT': 120
}

# Paths
PATHS = {
    'DATA_DIR': Path('data'),
    'CHECKPOINT_DIR': Path('checkpoints'),
    'OUTPUT_FILE': 'data/bls_tls_features.csv',
    'ERROR_LOG': 'logs/processing_errors.jsonl'
}

# Ensure directories exist
FOR path IN PATHS.values():
    IF path.parent exists:
        CREATE_DIRECTORY(path.parent)

PRINT "✅ Environment configured"


# ============================================================================
# CELL 3: Helper Functions
# ============================================================================

FUNCTION download_light_curve(tid, mission='TESS', max_retries=3):
    """
    Download light curve with retry logic

    Args:
        tid: TIC or KIC identifier
        mission: 'TESS' or 'Kepler'
        max_retries: Number of retry attempts

    Returns:
        LightCurve object or None
    """
    FOR attempt IN range(1, max_retries + 1):
        TRY:
            # Search for light curve
            search_result = lightkurve.search_lightcurve(
                tid,
                mission=mission,
                author='SPOC' IF mission=='TESS' ELSE 'Kepler'
            )

            IF search_result is empty:
                IF attempt < max_retries:
                    WAIT exponential_backoff(attempt)
                    CONTINUE
                ELSE:
                    RETURN None

            # Download first available sector/quarter
            lc = search_result[0].download()

            IF lc is None:
                IF attempt < max_retries:
                    WAIT exponential_backoff(attempt)
                    CONTINUE
                ELSE:
                    RETURN None

            RETURN lc

        CATCH Exception as e:
            LOG_ERROR(f"Download attempt {attempt} failed: {e}")
            IF attempt < max_retries:
                WAIT exponential_backoff(attempt)
            ELSE:
                RETURN None

    RETURN None


FUNCTION preprocess_light_curve(lc, window_length=401):
    """
    Clean and normalize light curve

    Args:
        lc: LightCurve object
        window_length: Flatten window size

    Returns:
        Preprocessed LightCurve
    """
    # Remove NaNs and Infs
    lc_clean = lc.remove_nans().remove_infinities()

    IF len(lc_clean) < CONFIG['MIN_POINTS']:
        RAISE PreprocessError("Too few points after cleaning")

    # Flatten (detrend)
    TRY:
        lc_flat = lc_clean.flatten(window_length=window_length)
    CATCH:
        # Fallback: simple detrending
        lc_flat = lc_clean - lc_clean.flux.median()
        lc_flat = lc_flat / lc_clean.flux.median()

    # Normalize to relative flux
    lc_normalized = lc_flat.normalize()

    # Clip outliers (5-sigma)
    median = np.median(lc_normalized.flux)
    std = np.std(lc_normalized.flux)
    mask = np.abs(lc_normalized.flux - median) < 5 * std
    lc_clipped = lc_normalized[mask]

    RETURN lc_clipped


FUNCTION run_bls_analysis(time, flux, period_range=(0.5, 20)):
    """
    Execute BLS period search

    Args:
        time: Time array (days)
        flux: Normalized flux array
        period_range: (min, max) period in days

    Returns:
        Dictionary with BLS results
    """
    # Create LightCurve object
    lc = lightkurve.LightCurve(time=time, flux=flux)

    # Run BLS
    bls = lc.to_periodogram(
        method='bls',
        minimum_period=period_range[0],
        maximum_period=period_range[1],
        minimum_n_transit=CONFIG['MIN_TRANSITS'],
        duration=np.linspace(
            CONFIG['DURATION_MIN'],
            CONFIG['DURATION_MAX'],
            20
        )
    )

    # Extract results
    results = {
        'bls_period': bls.period_at_max_power.value,
        'bls_power': bls.max_power,
        'bls_duration': bls.duration_at_max_power.value,
        'bls_depth': bls.depth_at_max_power.value,
        'bls_snr': calculate_snr(bls)
    }

    RETURN results


FUNCTION run_tls_analysis(time, flux, bls_period):
    """
    Execute TLS period search

    Args:
        time: Time array (days)
        flux: Normalized flux array
        bls_period: BLS period as initial guess

    Returns:
        Dictionary with TLS results
    """
    FROM transitleastsquares IMPORT transitleastsquares as TLS

    # Setup TLS
    model = TLS(time, flux)

    # Run TLS (use BLS period as reference)
    results = model.power(
        period_min=CONFIG['PERIOD_MIN'],
        period_max=CONFIG['PERIOD_MAX'],
        n_transits_min=CONFIG['MIN_TRANSITS']
    )

    # Extract results
    tls_results = {
        'tls_period': results.period,
        'tls_power': results.power.max(),
        'tls_duration': results.duration * 24,  # Convert to hours
        'tls_depth': results.depth * 1e6,  # Convert to ppm
        'tls_snr': results.snr,
        'tls_sde': results.SDE
    }

    RETURN tls_results


FUNCTION extract_features(lc, bls_results, tls_results, catalog_params):
    """
    Extract comprehensive feature set

    Args:
        lc: Preprocessed light curve
        bls_results: Dictionary from run_bls_analysis()
        tls_results: Dictionary from run_tls_analysis()
        catalog_params: Input parameters from TOI/KOI

    Returns:
        Dictionary with 27 features
    """
    time = lc.time.value
    flux = lc.flux.value

    features = {}

    # 1. Basic Statistics (4 features)
    features['flux_mean'] = np.mean(flux)
    features['flux_std'] = np.std(flux)
    features['flux_median'] = np.median(flux)
    features['flux_mad'] = median_absolute_deviation(flux)

    # 2. Input Parameters (4 features)
    features['input_period'] = catalog_params.get('pl_orbper', np.nan)
    features['input_duration'] = catalog_params.get('pl_trandurh', np.nan)
    features['input_depth'] = catalog_params.get('pl_trandep', np.nan)
    features['input_epoch'] = catalog_params.get('pl_tranmid', np.nan)

    # 3. BLS Features (5 features)
    features.update(bls_results)

    # 4. TLS Features (6 features)
    features.update(tls_results)

    # 5. Advanced Features (8 features)
    period = bls_results['bls_period']

    # Fold light curve
    folded_phase = ((time - time[0]) % period) / period
    folded_flux = flux

    # Odd-even mismatch
    odd_transits = extract_transits(time, flux, period, offset=0)
    even_transits = extract_transits(time, flux, period, offset=period)
    features['odd_even_mismatch'] = calculate_depth_difference(
        odd_transits, even_transits
    )

    # Secondary peak ratio
    features['secondary_power_ratio'] = calculate_secondary_peak_ratio(
        time, flux, period
    )

    # Harmonic rejection
    features['harmonic_delta_chisq'] = test_harmonic_periods(
        time, flux, period
    )

    # Periodicity strength
    features['periodicity_strength'] = calculate_phase_coherence(
        folded_phase, folded_flux
    )

    # Transit symmetry
    features['transit_symmetry'] = calculate_transit_symmetry(
        folded_phase, folded_flux
    )

    # Additional metrics
    features['odd_even_depth_diff'] = abs(
        features['odd_even_mismatch']
    ) * 1e6  # ppm

    features['phase_coverage'] = calculate_phase_coverage(
        folded_phase
    )

    features['ingress_egress_asymmetry'] = calculate_ingress_egress_ratio(
        folded_phase, folded_flux
    )

    RETURN features


FUNCTION save_checkpoint(features_df, index, config):
    """
    Save processing checkpoint

    Args:
        features_df: DataFrame with extracted features
        index: Current processing index
        config: Configuration dictionary
    """
    checkpoint_data = {
        'version': '1.0',
        'timestamp': datetime.now().isoformat(),
        'last_index': index,
        'total_processed': len(features_df),
        'features': features_df.to_dict('records'),
        'config': config
    }

    checkpoint_path = PATHS['CHECKPOINT_DIR'] / f'bls_checkpoint_{index:05d}.pkl'

    WITH open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)

    PRINT f"💾 Checkpoint saved: {checkpoint_path}"

    # Cleanup old checkpoints (keep last 3)
    cleanup_old_checkpoints(keep_last=3)


FUNCTION load_checkpoint():
    """
    Load most recent checkpoint

    Returns:
        Checkpoint dictionary or None
    """
    checkpoint_files = list(PATHS['CHECKPOINT_DIR'].glob('bls_checkpoint_*.pkl'))

    IF not checkpoint_files:
        RETURN None

    # Get most recent checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)

    PRINT f"📂 Loading checkpoint: {latest_checkpoint}"

    WITH open(latest_checkpoint, 'rb') as f:
        checkpoint_data = pickle.load(f)

    RETURN checkpoint_data


# ============================================================================
# CELL 4: Main Processing Function
# ============================================================================

FUNCTION process_single_sample(row, index):
    """
    Process a single sample

    Args:
        row: DataFrame row with sample data
        index: Sample index

    Returns:
        Dictionary with extracted features
    """
    tid = row['tid']

    TRY:
        # Step 1: Download light curve
        PRINT f"  [{index+1}] Downloading {tid}..."
        lc = download_light_curve(
            tid,
            mission='TESS' IF 'TIC' in tid ELSE 'Kepler',
            max_retries=CONFIG['MAX_RETRIES']
        )

        IF lc is None:
            RAISE DownloadError(f"Failed to download {tid}")

        # Step 2: Preprocess
        lc_processed = preprocess_light_curve(lc, CONFIG['FLATTEN_WINDOW'])

        # Step 3: BLS Analysis
        PRINT f"  [{index+1}] Running BLS for {tid}..."
        bls_results = run_bls_analysis(
            lc_processed.time.value,
            lc_processed.flux.value,
            period_range=(CONFIG['PERIOD_MIN'], CONFIG['PERIOD_MAX'])
        )

        # Step 4: TLS Analysis
        PRINT f"  [{index+1}] Running TLS for {tid}..."
        tls_results = run_tls_analysis(
            lc_processed.time.value,
            lc_processed.flux.value,
            bls_results['bls_period']
        )

        # Step 5: Feature Extraction
        catalog_params = {
            'pl_orbper': row.get('pl_orbper', np.nan),
            'pl_trandep': row.get('pl_trandep', np.nan),
            'pl_trandurh': row.get('pl_trandurh', np.nan),
            'pl_tranmid': row.get('pl_tranmid', np.nan)
        }

        features = extract_features(
            lc_processed,
            bls_results,
            tls_results,
            catalog_params
        )

        # Add metadata
        features['tid'] = tid
        features['toi'] = row.get('toi', np.nan)
        features['label'] = row.get('label', np.nan)
        features['success_flag'] = True
        features['error_message'] = None

        RETURN features

    CATCH DownloadError as e:
        LOG_ERROR(index, tid, 'DownloadError', str(e))
        RETURN create_failed_sample(tid, row, str(e))

    CATCH PreprocessError as e:
        LOG_ERROR(index, tid, 'PreprocessError', str(e))
        RETURN create_failed_sample(tid, row, str(e))

    CATCH Exception as e:
        LOG_ERROR(index, tid, 'UnexpectedError', str(e))
        RETURN create_failed_sample(tid, row, str(e))


FUNCTION create_failed_sample(tid, row, error_message):
    """Create feature dict for failed sample with NaN values"""
    features = {col: np.nan FOR col IN FEATURE_COLUMNS}
    features['tid'] = tid
    features['toi'] = row.get('toi', np.nan)
    features['label'] = row.get('label', np.nan)
    features['success_flag'] = False
    features['error_message'] = error_message
    RETURN features


# ============================================================================
# CELL 5: Main Processing Loop
# ============================================================================

PRINT "🚀 Starting BLS/TLS Feature Extraction"
PRINT "=" * 60

# Load dataset
PRINT "📂 Loading supervised dataset..."
df = pd.read_csv(PATHS['DATA_DIR'] / 'supervised_dataset.csv')
PRINT f"   Loaded {len(df)} samples"

# Check for existing checkpoint
checkpoint = load_checkpoint()

IF checkpoint is not None:
    PRINT f"📂 Resuming from checkpoint (index {checkpoint['last_index']})"
    features_list = checkpoint['features']
    start_index = checkpoint['last_index']
ELSE:
    PRINT "🆕 Starting fresh (no checkpoint found)"
    features_list = []
    start_index = 0

# Progress tracking
total_samples = len(df)
processed_count = start_index
success_count = len([f FOR f IN features_list IF f['success_flag']])
failed_count = len(features_list) - success_count

# Main processing loop
PRINT f"\n🔄 Processing {total_samples - start_index} remaining samples..."
PRINT f"   Batch size: {CONFIG['BATCH_SIZE']}"
PRINT f"   Checkpoint frequency: {CONFIG['CHECKPOINT_FREQ']} samples"
PRINT ""

start_time = time.time()

FOR batch_start IN range(start_index, total_samples, CONFIG['BATCH_SIZE']):
    batch_end = min(batch_start + CONFIG['BATCH_SIZE'], total_samples)

    PRINT f"\n📦 Batch {batch_start//CONFIG['BATCH_SIZE'] + 1}: "
    PRINT f"   Processing samples {batch_start} to {batch_end-1}"

    # Process batch with progress bar
    FOR index IN tqdm(range(batch_start, batch_end)):
        row = df.iloc[index]

        # Process sample
        features = process_single_sample(row, index)
        features_list.append(features)

        # Update counters
        processed_count += 1
        IF features['success_flag']:
            success_count += 1
        ELSE:
            failed_count += 1

        # Progress update
        IF (index + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = processed_count / elapsed
            remaining = (total_samples - processed_count) / rate

            PRINT f"\n  ⏱️ Progress: {processed_count}/{total_samples} "
            PRINT f"     ({100*processed_count/total_samples:.1f}%)"
            PRINT f"     Success: {success_count}, Failed: {failed_count}"
            PRINT f"     Rate: {rate:.1f} samples/min"
            PRINT f"     ETA: {remaining/3600:.1f} hours"

    # Save checkpoint
    IF (batch_end % CONFIG['CHECKPOINT_FREQ'] == 0) OR (batch_end == total_samples):
        features_df = pd.DataFrame(features_list)
        save_checkpoint(features_df, batch_end, CONFIG)

    # Memory cleanup
    import gc
    gc.collect()

# Final statistics
total_time = time.time() - start_time
PRINT "\n" + "=" * 60
PRINT "✅ Processing Complete!"
PRINT "=" * 60
PRINT f"📊 Summary:"
PRINT f"   Total samples: {total_samples}"
PRINT f"   Successful: {success_count} ({100*success_count/total_samples:.1f}%)"
PRINT f"   Failed: {failed_count} ({100*failed_count/total_samples:.1f}%)"
PRINT f"   Total time: {total_time/3600:.2f} hours"
PRINT f"   Average time: {total_time/total_samples:.1f} seconds/sample"
PRINT "=" * 60


# ============================================================================
# CELL 6: Save Final Results
# ============================================================================

PRINT "\n💾 Saving final results..."

# Convert to DataFrame
features_df = pd.DataFrame(features_list)

# Reorder columns
column_order = [
    # Metadata
    'tid', 'toi', 'label', 'success_flag',

    # Basic statistics
    'flux_mean', 'flux_std', 'flux_median', 'flux_mad',

    # Input parameters
    'input_period', 'input_duration', 'input_depth', 'input_epoch',

    # BLS features
    'bls_power', 'bls_period', 'bls_duration', 'bls_depth', 'bls_snr',

    # TLS features
    'tls_power', 'tls_period', 'tls_duration', 'tls_depth', 'tls_snr', 'tls_sde',

    # Advanced features
    'odd_even_mismatch', 'secondary_power_ratio', 'harmonic_delta_chisq',
    'periodicity_strength', 'transit_symmetry', 'odd_even_depth_diff',
    'phase_coverage', 'ingress_egress_asymmetry',

    # Error tracking
    'error_message'
]

features_df = features_df[column_order]

# Save to CSV
output_path = PATHS['OUTPUT_FILE']
features_df.to_csv(output_path, index=False)

PRINT f"✅ Results saved to: {output_path}"
PRINT f"   Shape: {features_df.shape}"
PRINT f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB"

# Save metadata
metadata = {
    'generation_date': datetime.now().isoformat(),
    'total_samples': len(features_df),
    'successful_samples': success_count,
    'failed_samples': failed_count,
    'success_rate': success_count / len(features_df),
    'processing_time_hours': total_time / 3600,
    'average_time_per_sample_seconds': total_time / len(features_df),
    'config': CONFIG,
    'numpy_version': np.__version__,
    'lightkurve_version': lightkurve.__version__
}

metadata_path = PATHS['DATA_DIR'] / 'bls_features_metadata.json'
WITH open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

PRINT f"✅ Metadata saved to: {metadata_path}"


# ============================================================================
# CELL 7: Visualization and Analysis
# ============================================================================

PRINT "\n📊 Generating visualizations..."

import matplotlib.pyplot as plt
import seaborn as sns

# Filter successful samples
success_df = features_df[features_df['success_flag'] == True]

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Success rate
ax1 = axes[0, 0]
success_rate = [success_count, failed_count]
ax1.pie(success_rate, labels=['Success', 'Failed'], autopct='%1.1f%%')
ax1.set_title(f'Processing Success Rate\n({success_count}/{total_samples})')

# 2. BLS Period distribution
ax2 = axes[0, 1]
IF len(success_df) > 0:
    ax2.hist(success_df['bls_period'], bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('BLS Period (days)')
    ax2.set_ylabel('Count')
    ax2.set_title('BLS Period Distribution')
    ax2.set_xlim(CONFIG['PERIOD_MIN'], CONFIG['PERIOD_MAX'])

# 3. BLS SNR distribution
ax3 = axes[0, 2]
IF len(success_df) > 0:
    ax3.hist(success_df['bls_snr'], bins=50, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('BLS SNR')
    ax3.set_ylabel('Count')
    ax3.set_title('BLS SNR Distribution')
    ax3.axvline(x=10, color='red', linestyle='--', label='SNR=10 threshold')
    ax3.legend()

# 4. Transit depth distribution
ax4 = axes[1, 0]
IF len(success_df) > 0:
    depths_ppm = success_df['bls_depth'] * 1e6
    ax4.hist(depths_ppm, bins=50, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Transit Depth (ppm)')
    ax4.set_ylabel('Count')
    ax4.set_title('Transit Depth Distribution')
    ax4.set_xlim(0, 10000)

# 5. Period vs Depth
ax5 = axes[1, 1]
IF len(success_df) > 0:
    scatter = ax5.scatter(
        success_df['bls_period'],
        success_df['bls_depth'] * 1e6,
        c=success_df['bls_snr'],
        cmap='viridis',
        alpha=0.6,
        s=20
    )
    ax5.set_xlabel('Period (days)')
    ax5.set_ylabel('Depth (ppm)')
    ax5.set_title('Period vs Depth (colored by SNR)')
    plt.colorbar(scatter, ax=ax5, label='SNR')
    ax5.set_xscale('log')
    ax5.set_yscale('log')

# 6. Label distribution (if available)
ax6 = axes[1, 2]
IF 'label' IN success_df.columns:
    label_counts = success_df['label'].value_counts()
    ax6.bar(['False Positive', 'Planet'], label_counts.values)
    ax6.set_ylabel('Count')
    ax6.set_title('Label Distribution (Successful Samples)')

plt.tight_layout()
plt.savefig('figures/bls_baseline_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

PRINT "✅ Visualizations complete"


# ============================================================================
# CELL 8: GitHub Push (Optional)
# ============================================================================

PRINT "\n📤 Ready to push results to GitHub"
PRINT "💡 Uncomment and run the following line when ready:"
PRINT "   ultimate_push_to_github_02()"

# ultimate_push_to_github_02()
```

---

## 2. Notebook 03: Supervised Learning Training

### 2.1 Specification

#### Purpose
Train machine learning classifiers (Logistic Regression, Random Forest, XGBoost) on extracted BLS/TLS features to predict exoplanet candidates. Includes probability calibration, cross-validation, and model persistence.

#### Functional Requirements

**FR-3.1: Data Loading**
- **ID**: FR-3.1
- **Description**: Load and merge feature dataset
- **Priority**: Critical
- **Acceptance Criteria**:
  - Load `bls_tls_features.csv` (11,979 rows)
  - Merge with `supervised_dataset.csv` for labels
  - Handle missing labels gracefully
  - Filter samples with `success_flag == True`
  - Minimum samples after filtering: 10,000+

**FR-3.2: Feature Engineering**
- **ID**: FR-3.2
- **Description**: Prepare features for training
- **Priority**: High
- **Acceptance Criteria**:
  - Drop metadata columns (`tid`, `toi`, `error_message`)
  - Handle missing values (imputation or removal)
  - Feature scaling (StandardScaler or RobustScaler)
  - Feature selection (optional, remove low-variance)
  - Train/test split: 80/20 with stratification

**FR-3.3: Model Training - Logistic Regression**
- **ID**: FR-3.3
- **Description**: Train baseline logistic regression
- **Priority**: High
- **Acceptance Criteria**:
  - Regularization: L2 (Ridge)
  - Cross-validation: 5-fold StratifiedGroupKFold
  - Hyperparameters: C ∈ [0.001, 0.01, 0.1, 1, 10]
  - Training time: < 2 minutes
  - ROC-AUC > 0.85 on validation

**FR-3.4: Model Training - Random Forest**
- **ID**: FR-3.4
- **Description**: Train ensemble model
- **Priority**: Medium
- **Acceptance Criteria**:
  - Number of trees: 100-500
  - Max depth: [10, 20, None]
  - Min samples split: [2, 5, 10]
  - Class weights: balanced
  - Training time: < 5 minutes
  - ROC-AUC > 0.90 on validation

**FR-3.5: Model Training - XGBoost with GPU**
- **ID**: FR-3.5
- **Description**: Train gradient boosting model with GPU acceleration
- **Priority**: Critical
- **Acceptance Criteria**:
  - Use `device='cuda'` if GPU available
  - Number of estimators: 100-300
  - Learning rate: [0.01, 0.05, 0.1]
  - Max depth: [3, 5, 7]
  - Early stopping: 20 rounds
  - Training time: < 10 minutes with GPU, < 30 minutes CPU
  - ROC-AUC > 0.92 on validation

**FR-3.6: Probability Calibration**
- **ID**: FR-3.6
- **Description**: Calibrate predicted probabilities
- **Priority**: High
- **Acceptance Criteria**:
  - Method: Isotonic Regression
  - Alternative: Platt Scaling
  - Calibration on validation set
  - Generate calibration curve
  - Expected Calibration Error (ECE) < 0.10

**FR-3.7: Model Evaluation**
- **ID**: FR-3.7
- **Description**: Comprehensive model assessment
- **Priority**: High
- **Acceptance Criteria**:
  - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
  - Confusion matrix visualization
  - Precision-Recall curve
  - ROC curve
  - Feature importance plot (for tree models)
  - Classification report

**FR-3.8: Model Persistence**
- **ID**: FR-3.8
- **Description**: Save trained models and artifacts
- **Priority**: Critical
- **Acceptance Criteria**:
  - Save best model: `models/best_model.joblib`
  - Save scaler: `models/scaler.joblib`
  - Save feature schema: `models/feature_schema.json`
  - Save calibrator: `models/calibrator.joblib`
  - Save training metrics: `models/training_report.json`
  - Total size: < 50 MB

#### Non-Functional Requirements

**NFR-3.1: Performance**
- **ID**: NFR-3.1
- **Category**: Performance
- **Description**: Efficient training with GPU acceleration
- **Measurement**:
  - XGBoost training with GPU: 5-10 minutes
  - XGBoost training without GPU: < 30 minutes
  - Memory usage: < 16GB peak
  - GPU memory: < 4GB (T4 compatible)

**NFR-3.2: Reproducibility**
- **ID**: NFR-3.2
- **Category**: Reproducibility
- **Description**: Results must be reproducible
- **Validation**:
  - Random seed set: 42
  - Cross-validation seed fixed
  - Model parameters logged
  - Environment versions recorded

**NFR-3.3: Model Quality**
- **ID**: NFR-3.3
- **Category**: Quality
- **Description**: Minimum performance thresholds
- **Validation**:
  - XGBoost ROC-AUC ≥ 0.92
  - Precision at 80% recall ≥ 0.85
  - False positive rate ≤ 15%
  - Calibration ECE ≤ 0.10

#### Use Cases

**UC-3.1: Full Pipeline Training**
```gherkin
Feature: Train complete ML pipeline

Scenario: Train from scratch with GPU
  Given bls_tls_features.csv exists with 11,979 samples
  And GPU is available (T4/A100)
  When user runs training notebook
  Then system loads and preprocesses data
  And trains 3 models (LogReg, RF, XGBoost)
  And calibrates probabilities
  And saves best model (XGBoost)
  And training completes in < 15 minutes
  And final ROC-AUC > 0.92
```

**UC-3.2: Model Comparison**
```gherkin
Feature: Compare multiple models

Scenario: Evaluate model performance
  Given 3 trained models exist
  When user generates comparison report
  Then system displays metrics table
  And shows ROC curves for all models
  And shows PR curves for all models
  And recommends best model
  And exports comparison to PDF
```

### 2.2 Training Architecture

```yaml
# Training Pipeline Architecture

pipeline:
  name: "Exoplanet Classifier Training"
  version: "1.0"

  stages:
    - name: "Data Loading"
      inputs:
        - data/bls_tls_features.csv
        - data/supervised_dataset.csv
      outputs:
        - X_train, X_val, y_train, y_val

    - name: "Preprocessing"
      steps:
        - Remove failed samples
        - Handle missing values
        - Feature scaling (StandardScaler)
        - Train/validation split (80/20)
      outputs:
        - scaler object
        - feature schema

    - name: "Model Training"
      models:
        logistic_regression:
          hyperparameters:
            C: [0.001, 0.01, 0.1, 1, 10]
            penalty: ['l2']
            solver: ['lbfgs']
          cv: 5-fold StratifiedKFold

        random_forest:
          hyperparameters:
            n_estimators: [100, 200, 500]
            max_depth: [10, 20, None]
            min_samples_split: [2, 5, 10]
            class_weight: ['balanced']
          cv: 5-fold StratifiedKFold

        xgboost:
          hyperparameters:
            n_estimators: [100, 200, 300]
            learning_rate: [0.01, 0.05, 0.1]
            max_depth: [3, 5, 7]
            subsample: [0.8, 1.0]
            colsample_bytree: [0.8, 1.0]
          gpu: true
          early_stopping_rounds: 20
          cv: 5-fold StratifiedKFold

    - name: "Calibration"
      method: "IsotonicRegression"
      cv_folds: 5

    - name: "Evaluation"
      metrics:
        - accuracy
        - precision
        - recall
        - f1_score
        - roc_auc
        - pr_auc
        - brier_score
        - expected_calibration_error

    - name: "Model Persistence"
      artifacts:
        - models/best_model.joblib
        - models/scaler.joblib
        - models/calibrator.joblib
        - models/feature_schema.json
        - models/training_report.json
```

### 2.3 Implementation Roadmap

```python
# Cell Structure for Notebook 03

# CELL 1: Package Installation & GPU Setup
"""
- Install XGBoost with GPU support
- Install sklearn, joblib, matplotlib
- Detect GPU availability
"""

# CELL 2: Load Data
"""
- Load bls_tls_features.csv
- Filter successful samples
- Prepare X (features) and y (labels)
"""

# CELL 3: Preprocessing
"""
- Feature selection
- Train/test split
- Scaling (StandardScaler)
"""

# CELL 4: Baseline - Logistic Regression
"""
- Train with GridSearchCV
- Evaluate on validation set
- Save model
"""

# CELL 5: Random Forest
"""
- Train with GridSearchCV
- Feature importance analysis
- Evaluate and save
"""

# CELL 6: XGBoost (GPU Accelerated)
"""
- Configure XGBoost with device='cuda'
- Train with early stopping
- Plot training curves
- Evaluate and save
"""

# CELL 7: Probability Calibration
"""
- Apply Isotonic Regression
- Generate calibration curves
- Compare before/after ECE
"""

# CELL 8: Model Comparison
"""
- Generate comparison table
- Plot ROC curves (all models)
- Plot PR curves (all models)
- Recommend best model
"""

# CELL 9: Save Best Model
"""
- Export XGBoost pipeline
- Save scaler and calibrator
- Generate training report
"""

# CELL 10: Visualizations
"""
- Confusion matrix
- Feature importance
- Calibration plot
- Error analysis
"""
```

---

## 3. Notebook 04: New Data Inference Pipeline

### 3.1 Specification

#### Purpose
Provide end-to-end inference pipeline for predicting exoplanet candidates from new TIC identifiers. Supports single-target and batch processing with GPU optimization.

#### Functional Requirements

**FR-4.1: Model Loading**
- **ID**: FR-4.1
- **Description**: Load trained model and artifacts
- **Priority**: Critical
- **Acceptance Criteria**:
  - Load XGBoost model from `models/best_model.joblib`
  - Load scaler from `models/scaler.joblib`
  - Load calibrator (if exists)
  - Validate model compatibility
  - Load time: < 5 seconds

**FR-4.2: Single Target Inference**
- **ID**: FR-4.2
- **Description**: Predict for single TIC identifier
- **Priority**: High
- **Acceptance Criteria**:
  - Input: TIC ID (e.g., "TIC 25155310")
  - Download light curve from MAST
  - Extract BLS/TLS features
  - Apply model inference
  - Return: probability, period, depth, SNR
  - Total time: < 60 seconds per target

**FR-4.3: Batch Inference**
- **ID**: FR-4.3
- **Description**: Process multiple targets efficiently
- **Priority**: High
- **Acceptance Criteria**:
  - Input: List of TIC IDs
  - Parallel download (max 5 concurrent)
  - Batch feature extraction
  - Batch prediction
  - Return: DataFrame with results sorted by probability
  - Average time: < 45 seconds per target

**FR-4.4: Visualization**
- **ID**: FR-4.4
- **Description**: Generate diagnostic plots
- **Priority**: Medium
- **Acceptance Criteria**:
  - Folded light curve plot
  - BLS power spectrum
  - Prediction confidence bar
  - Feature value summary
  - Phase coverage plot
  - Transit shape visualization

**FR-4.5: Result Export**
- **ID**: FR-4.5
- **Description**: Save inference results
- **Priority**: Medium
- **Acceptance Criteria**:
  - CSV format with standardized columns
  - Include metadata (mission, sector, date)
  - Include provenance (model version, features used)
  - Human-readable format
  - Machine-readable JSON metadata

**FR-4.6: GPU Optimization**
- **ID**: FR-4.6
- **Description**: Leverage GPU for acceleration
- **Priority**: Low (optional)
- **Acceptance Criteria**:
  - Detect L4 GPU automatically
  - Use BFloat16 autocast if available
  - Batch inference with GPU
  - Measure speedup (target: 2-3x)

#### Non-Functional Requirements

**NFR-4.1: Latency**
- **ID**: NFR-4.1
- **Category**: Performance
- **Description**: Low-latency predictions
- **Measurement**:
  - Single target: < 60 seconds
  - Batch (10 targets): < 10 minutes
  - GPU batch (10 targets): < 5 minutes

**NFR-4.2: Reliability**
- **ID**: NFR-4.2
- **Category**: Reliability
- **Description**: Robust error handling
- **Validation**:
  - Handle network failures gracefully
  - Continue batch on individual failures
  - Provide clear error messages
  - Log all errors with timestamps

#### Use Cases

**UC-4.1: Quick Check - Single Target**
```gherkin
Feature: Check single candidate

Scenario: User wants to check one TIC
  Given model is loaded
  When user provides TIC 25155310
  Then system downloads light curve
  And extracts features
  And returns probability 0.875
  And displays folded light curve
  And shows period 4.178 days
```

**UC-4.2: Batch Screening**
```gherkin
Feature: Screen multiple candidates

Scenario: User has list of 50 TICs
  Given model is loaded
  And user provides list of 50 TICs
  When user runs batch inference
  Then system processes all targets
  And returns sorted DataFrame
  And saves results to CSV
  And displays top 10 candidates
  And batch completes in < 40 minutes
```

### 3.2 Inference Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Notebook 04: Inference Pipeline                │
└─────────────────────────────────────────────────────────────────┘

INPUT:
  TIC ID(s) → "TIC 25155310" or ["TIC 123", "TIC 456", ...]

PIPELINE:

  ┌──────────────────────┐
  │ 1. Model Loading     │
  │  - Load XGBoost      │
  │  - Load scaler       │
  │  - Load calibrator   │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ 2. Light Curve       │
  │    Download          │
  │  - Query MAST        │
  │  - Fetch TESS data   │
  │  - Retry logic       │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ 3. Preprocessing     │
  │  - Remove NaNs       │
  │  - Flatten curve     │
  │  - Normalize         │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ 4. Feature           │
  │    Extraction        │
  │  - BLS analysis      │
  │  - TLS analysis      │
  │  - 27 features       │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ 5. Inference         │
  │  - Scale features    │
  │  - XGBoost predict   │
  │  - Calibrate prob    │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │ 6. Visualization     │
  │  - Folded LC plot    │
  │  - BLS power spec    │
  │  - Confidence bars   │
  └──────────┬───────────┘
             │
             ▼
OUTPUT:
  results_df:
    - tic_id
    - probability
    - bls_period
    - bls_depth
    - bls_snr
    - success_flag
```

### 3.3 Implementation Skeleton

```python
# Key functions for Notebook 04

def load_inference_artifacts():
    """Load model, scaler, calibrator"""
    model = joblib.load('models/best_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    calibrator = joblib.load('models/calibrator.joblib')
    return model, scaler, calibrator

def predict_single_target(tic_id, model, scaler, calibrator):
    """
    Full inference pipeline for one target

    Returns:
        {
            'tic_id': str,
            'probability': float,
            'bls_period': float,
            'bls_snr': float,
            'success': bool
        }
    """
    # 1. Download light curve
    lc = download_light_curve(tic_id)

    # 2. Preprocess
    lc_clean = preprocess_light_curve(lc)

    # 3. Extract features
    features = extract_all_features(lc_clean)

    # 4. Scale features
    features_scaled = scaler.transform([features])

    # 5. Predict
    prob_raw = model.predict_proba(features_scaled)[0, 1]

    # 6. Calibrate
    prob_calibrated = calibrator.predict([prob_raw])[0]

    return {
        'tic_id': tic_id,
        'probability': prob_calibrated,
        'bls_period': features['bls_period'],
        'bls_snr': features['bls_snr'],
        'success': True
    }

def predict_batch(tic_list, model, scaler, calibrator):
    """Batch inference with parallel downloads"""
    results = []

    for tic in tqdm(tic_list):
        try:
            result = predict_single_target(tic, model, scaler, calibrator)
            results.append(result)
        except Exception as e:
            results.append({
                'tic_id': tic,
                'success': False,
                'error': str(e)
            })

    df = pd.DataFrame(results)
    df = df.sort_values('probability', ascending=False)
    return df

def visualize_candidate(tic_id, result):
    """Generate diagnostic plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Light curve
    # Plot 2: Folded curve
    # Plot 3: BLS power
    # Plot 4: Probability bar

    plt.tight_layout()
    plt.show()
```

---

## 4. Validation & Testing

### 4.1 Notebook 02 Validation

**Unit Tests:**
- Download function returns valid LightCurve
- Preprocessing removes NaNs correctly
- BLS extracts non-zero period
- TLS improves on BLS period
- Feature extraction returns 27 values

**Integration Tests:**
- Full pipeline processes one sample successfully
- Checkpoint saves and resumes correctly
- Error handling continues after failure
- Final CSV has correct shape and columns

**Acceptance Tests:**
- Process 100 samples successfully
- Success rate > 90%
- Average time < 50 seconds per sample
- Output CSV validates against schema

### 4.2 Notebook 03 Validation

**Unit Tests:**
- Data loading handles missing values
- Feature scaling transforms correctly
- XGBoost trains with GPU
- Calibration improves ECE
- Model saves and loads successfully

**Integration Tests:**
- Full training pipeline completes
- Cross-validation produces consistent results
- Model comparison generates valid report
- Best model meets ROC-AUC threshold

**Acceptance Tests:**
- XGBoost ROC-AUC ≥ 0.92
- Training time < 15 minutes with GPU
- ECE after calibration < 0.10
- Model exports < 50 MB total size

### 4.3 Notebook 04 Validation

**Unit Tests:**
- Model loading succeeds
- Single target inference returns probability
- Batch inference handles list of TICs
- Visualization generates valid plots
- Export produces valid CSV

**Integration Tests:**
- End-to-end single target pipeline
- End-to-end batch pipeline
- GPU acceleration provides speedup
- Error handling recovers from failures

**Acceptance Tests:**
- Single target < 60 seconds
- Batch of 10 targets < 10 minutes
- All outputs are reproducible
- Results match expected format

---

## 5. Deliverables Summary

### 5.1 Notebook 02 Deliverables
- ✅ Working notebook in Google Colab
- ✅ Output: `data/bls_tls_features.csv` (11,979 rows × 31 columns)
- ✅ Checkpoint system (resume capability)
- ✅ Processing time: 20-30 hours
- ✅ Success rate: > 90%
- ✅ Visualization: Feature distributions
- ✅ Documentation: Runtime restart instructions

### 5.2 Notebook 03 Deliverables
- ✅ Working notebook in Google Colab
- ✅ Trained models: LogReg, RF, XGBoost
- ✅ Best model: `models/best_model.joblib`
- ✅ Artifacts: scaler, calibrator, feature schema
- ✅ Training report: metrics, confusion matrix
- ✅ XGBoost ROC-AUC: ≥ 0.92
- ✅ Training time: < 15 minutes with GPU
- ✅ Calibration: ECE < 0.10

### 5.3 Notebook 04 Deliverables
- ✅ Working notebook in Google Colab
- ✅ Single target inference: < 60 seconds
- ✅ Batch inference: functional and tested
- ✅ Visualizations: 6 diagnostic plots
- ✅ Export: Standardized CSV with metadata
- ✅ GPU optimization: BFloat16 support
- ✅ Error handling: Robust and informative
- ✅ Documentation: Usage examples

---

## 6. Success Criteria

### Overall Project Success
- [x] Dataset: 11,979 samples available
- [ ] Notebook 02: Extract features for all samples (>90% success)
- [ ] Notebook 03: Train XGBoost with ROC-AUC ≥ 0.92
- [ ] Notebook 04: Inference pipeline < 60s per target
- [ ] All notebooks: Execute without errors in fresh Colab
- [ ] Documentation: Complete specifications (this document)
- [ ] GitHub: All code and results committed

### Quality Metrics
- **Data Quality**: >90% samples successfully processed
- **Model Performance**: XGBoost ROC-AUC ≥ 0.92, PR-AUC ≥ 0.85
- **Calibration**: ECE < 0.10
- **Inference Speed**: <60s single target, <10min for 10 targets
- **Reproducibility**: All results reproducible with seed=42
- **Documentation**: All specifications have acceptance criteria

---

## 7. Risk Mitigation

### Risk 1: Colab Disconnects During 20-Hour Processing
**Impact**: High
**Mitigation**:
- Implement robust checkpoint system (save every 100 samples)
- Use Google Drive for persistent storage
- Enable background execution
- Test resume capability thoroughly

### Risk 2: NumPy 2.0 Incompatibility
**Impact**: Critical
**Mitigation**:
- Force install NumPy 1.26.4 in first cell
- Add explicit runtime restart instruction
- Test with fresh Colab environment
- Document workaround in troubleshooting guide

### Risk 3: MAST Download Failures
**Impact**: Medium
**Mitigation**:
- Implement retry logic (3 attempts)
- Exponential backoff between retries
- Fallback to alternative sectors
- Continue processing on failure
- Log all failed downloads

### Risk 4: GPU Unavailable for Training
**Impact**: Medium
**Mitigation**:
- Detect GPU availability at runtime
- Fallback to CPU with clear warnings
- Optimize hyperparameters for CPU
- Document expected training times for both

### Risk 5: Model Overfitting
**Impact**: Medium
**Mitigation**:
- Use StratifiedKFold cross-validation
- Monitor train/validation curves
- Early stopping (20 rounds)
- Test on held-out test set
- Calibration to prevent overconfidence

---

## 8. Timeline Estimate

### Notebook 02 Development: 1-2 days
- Implementation: 4-6 hours
- Testing: 2-3 hours
- Documentation: 1-2 hours
- **Execution**: 20-30 hours (background)

### Notebook 03 Development: 1 day
- Implementation: 3-4 hours
- Training: 1 hour (with GPU)
- Evaluation: 1-2 hours
- Documentation: 1 hour

### Notebook 04 Development: 0.5 days
- Implementation: 2-3 hours
- Testing: 1-2 hours
- Documentation: 1 hour

### Total Development Time: 2.5-3.5 days
### Total Execution Time: 20-30 hours (mostly automated)

---

## 9. References

### NASA Data Sources
- [NASA Exoplanet Archive TAP Service](https://exoplanetarchive.ipac.caltech.edu/TAP)
- [TESS Objects of Interest (TOI)](https://exoplanetarchive.ipac.caltech.edu/docs/API_TOI_columns.html)
- [Kepler Objects of Interest (KOI)](https://exoplanetarchive.ipac.caltech.edu/docs/Kepler_KOI_docs.html)

### Technical Documentation
- [Lightkurve Documentation](https://docs.lightkurve.org/)
- [TransitLeastSquares](https://github.com/hippke/tls)
- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/stable/gpu/index.html)
- [Scikit-learn Calibration](https://scikit-learn.org/stable/modules/calibration.html)

### Research Papers
- Kovács et al. (2002) - BLS Algorithm
- Hippke & Heller (2019) - Transit Least Squares
- Chen & Guestrin (2016) - XGBoost
- Platt (1999) - Probability Calibration

---

**Document Status**: ✅ COMPLETE
**Next Action**: Begin implementation of Notebook 02
**Expected Completion**: 2025-10-05 (including 20-30 hour execution)

---

*Generated using SPARC methodology by Claude Code*
*Last updated: 2025-09-30*