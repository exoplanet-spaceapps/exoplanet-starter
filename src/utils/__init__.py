"""
Utility modules for Exoplanet Detection Pipeline
"""
from .reproducibility import set_random_seeds, get_random_state, restore_random_state
from .logging_config import setup_logger, get_log_file_path, log_system_info, log_data_info
from .gpu_utils import (
    detect_gpu,
    get_xgboost_gpu_params,
    log_gpu_info,
    get_pytorch_device,
    configure_gpu_memory_growth,
    print_gpu_memory_usage
)
from .model_card import create_model_card, save_model_card, load_model_card
from .provenance import (
    create_provenance_record,
    save_provenance,
    load_provenance,
    update_provenance
)
from .calibration_viz import plot_calibration_curves, compare_calibration_methods
from .output_schema import (
    create_candidate_dataframe,
    export_candidates_csv,
    export_candidates_jsonl
)
from .latency_metrics import (
    LatencyTimer,
    LatencyTracker,
    calculate_percentiles,
    calculate_latency_stats,
    plot_latency_histogram
)

# Optional imports - require additional packages
try:
    from .plotly_viz import (
        create_interactive_roc_curve,
        create_interactive_pr_curve,
        create_interactive_confusion_matrix,
        create_interactive_feature_importance,
        create_interactive_calibration_curve,
        create_metrics_dashboard,
        export_to_html
    )
    _PLOTLY_AVAILABLE = True
except ImportError:
    # Plotly not installed - interactive visualizations unavailable
    _PLOTLY_AVAILABLE = False

__all__ = [
    # Reproducibility
    'set_random_seeds',
    'get_random_state',
    'restore_random_state',
    # Logging
    'setup_logger',
    'get_log_file_path',
    'log_system_info',
    'log_data_info',
    # GPU
    'detect_gpu',
    'get_xgboost_gpu_params',
    'log_gpu_info',
    'get_pytorch_device',
    'configure_gpu_memory_growth',
    'print_gpu_memory_usage',
    # Model Card
    'create_model_card',
    'save_model_card',
    'load_model_card',
    # Provenance
    'create_provenance_record',
    'save_provenance',
    'load_provenance',
    'update_provenance',
    # Calibration Visualization
    'plot_calibration_curves',
    'compare_calibration_methods',
    # Output Schema
    'create_candidate_dataframe',
    'export_candidates_csv',
    'export_candidates_jsonl',
    # Latency Metrics
    'LatencyTimer',
    'LatencyTracker',
    'calculate_percentiles',
    'calculate_latency_stats',
    'plot_latency_histogram',
    # Plotly Visualization
    'create_interactive_roc_curve',
    'create_interactive_pr_curve',
    'create_interactive_confusion_matrix',
    'create_interactive_feature_importance',
    'create_interactive_calibration_curve',
    'create_metrics_dashboard',
    'export_to_html'
]