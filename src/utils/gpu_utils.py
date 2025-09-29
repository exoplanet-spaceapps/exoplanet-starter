"""
GPU Detection and Configuration Utilities
GPU 檢測與配置工具 (2025 Best Practices)
"""
import subprocess
import logging
from typing import Dict, Any, Optional


def detect_gpu() -> Dict[str, Any]:
    """
    檢測 GPU 可用性與配置

    Returns:
        dict: GPU 資訊字典，包含:
            - available: GPU 是否可用 (bool)
            - device_count: GPU 數量 (int)
            - device_name: GPU 名稱 (str)
            - cuda_version: CUDA 版本 (str)
            - memory_gb: 總記憶體 (float)
            - pytorch_available: PyTorch 是否可用 (bool)
            - xgboost_gpu_support: XGBoost GPU 是否支援 (bool)
    """
    gpu_info = {
        'available': False,
        'device_count': 0,
        'device_name': None,
        'cuda_version': None,
        'memory_gb': None,
        'pytorch_available': False,
        'xgboost_gpu_support': False
    }

    # 檢查 PyTorch GPU 支援
    try:
        import torch
        gpu_info['pytorch_available'] = True

        if torch.cuda.is_available():
            gpu_info['available'] = True
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
            gpu_info['cuda_version'] = torch.version.cuda
            gpu_info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1024**3

    except ImportError:
        pass

    # 如果 PyTorch 不可用，嘗試用 nvidia-smi
    if not gpu_info['available']:
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                check=False,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu_info['available'] = True
                gpu_info['device_name'] = result.stdout.strip()
                gpu_info['device_count'] = len(result.stdout.strip().split('\n'))
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # 檢查 XGBoost GPU 支援
    try:
        import xgboost as xgb
        # XGBoost 2.0+ 支援 GPU
        xgb_version = tuple(map(int, xgb.__version__.split('.')[:2]))
        if xgb_version >= (2, 0) and gpu_info['available']:
            gpu_info['xgboost_gpu_support'] = True
    except ImportError:
        pass

    return gpu_info


def get_xgboost_gpu_params(
    tree_method: str = 'hist',
    device: str = 'cuda',
    gpu_id: int = 0
) -> Dict[str, Any]:
    """
    獲取 XGBoost 2.x GPU 訓練參數

    Args:
        tree_method: 樹構建方法，推薦 'hist' (最快)
        device: 設備類型，'cuda' 或 'cpu'
        gpu_id: GPU ID (如果有多個 GPU)

    Returns:
        dict: XGBoost 參數字典

    Example:
        >>> gpu_params = get_xgboost_gpu_params()
        >>> model = xgb.XGBClassifier(**gpu_params, n_estimators=100)
    """
    gpu_info = detect_gpu()

    if not gpu_info['available'] or not gpu_info['xgboost_gpu_support']:
        return {
            'tree_method': 'hist',
            'device': 'cpu'
        }

    # XGBoost 2.x 使用 device='cuda' 而不是 gpu_id
    return {
        'tree_method': tree_method,  # 'hist' 是最快的方法
        'device': device,             # 'cuda' 啟用 GPU
        # 'gpu_id': gpu_id,           # 舊版用法，2.x 不需要
        # 'predictor': 'gpu_predictor', # 舊版用法，2.x 會自動使用
    }


def log_gpu_info(logger: Optional[logging.Logger] = None) -> None:
    """
    記錄 GPU 資訊到日誌

    Args:
        logger: Logger 物件，如果為 None 則使用 print
    """
    gpu_info = detect_gpu()

    if logger is None:
        print_fn = print
    else:
        print_fn = logger.info

    print_fn("="*60)
    print_fn("🖥️ GPU Configuration")
    print_fn("="*60)

    if gpu_info['available']:
        print_fn(f"GPU Available: ✅ YES")
        print_fn(f"GPU Count: {gpu_info['device_count']}")
        print_fn(f"GPU Name: {gpu_info['device_name']}")

        if gpu_info['cuda_version']:
            print_fn(f"CUDA Version: {gpu_info['cuda_version']}")

        if gpu_info['memory_gb']:
            print_fn(f"GPU Memory: {gpu_info['memory_gb']:.2f} GB")

        # L4 GPU 特殊提示
        if gpu_info['device_name'] and 'L4' in gpu_info['device_name']:
            print_fn("💡 NVIDIA L4 GPU Detected - Supports BF16 acceleration")

        # XGBoost GPU 支援
        if gpu_info['xgboost_gpu_support']:
            print_fn("XGBoost GPU: ✅ Supported (use device='cuda')")
        else:
            print_fn("XGBoost GPU: ❌ Not supported (need XGBoost 2.0+)")

    else:
        print_fn("GPU Available: ❌ NO - Using CPU")
        print_fn("💡 For faster training, use Colab with GPU runtime:")
        print_fn("   Runtime → Change runtime type → Hardware accelerator → GPU")

    print_fn("="*60)


def get_pytorch_device() -> str:
    """
    獲取 PyTorch 最佳設備

    Returns:
        str: 'cuda' 或 'cpu'

    Example:
        >>> device = get_pytorch_device()
        >>> model = model.to(device)
    """
    gpu_info = detect_gpu()
    return 'cuda' if gpu_info['available'] and gpu_info['pytorch_available'] else 'cpu'


def configure_gpu_memory_growth():
    """
    配置 TensorFlow/PyTorch 動態記憶體增長
    避免一次性占用所有 GPU 記憶體
    """
    # TensorFlow
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ TensorFlow GPU memory growth enabled ({len(gpus)} GPUs)")
    except (ImportError, RuntimeError):
        pass

    # PyTorch (不需要特別配置，預設就是動態分配)
    try:
        import torch
        if torch.cuda.is_available():
            # 設定 PyTorch CUDA 記憶體分配策略
            torch.cuda.empty_cache()  # 清空快取
            print(f"✅ PyTorch GPU available ({torch.cuda.device_count()} GPUs)")
    except ImportError:
        pass


def print_gpu_memory_usage():
    """
    顯示當前 GPU 記憶體使用情況
    """
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3

                print(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
                print(f"  Allocated: {allocated:.2f} GB / {total:.2f} GB")
                print(f"  Reserved:  {reserved:.2f} GB / {total:.2f} GB")
                print(f"  Free:      {total - allocated:.2f} GB")
    except ImportError:
        print("PyTorch not available - cannot check GPU memory")