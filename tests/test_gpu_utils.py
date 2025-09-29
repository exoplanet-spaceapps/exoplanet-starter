"""
Tests for GPU utilities
"""
# Fix UTF-8 encoding for Windows
import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from utils.gpu_utils import detect_gpu, get_xgboost_gpu_params, get_pytorch_device


def test_detect_gpu():
    """Test GPU detection"""
    gpu_info = detect_gpu()

    # Check all expected keys exist
    assert 'available' in gpu_info
    assert 'device_count' in gpu_info
    assert 'device_name' in gpu_info
    assert 'cuda_version' in gpu_info
    assert 'memory_gb' in gpu_info
    assert 'pytorch_available' in gpu_info
    assert 'xgboost_gpu_support' in gpu_info

    # Check types
    assert isinstance(gpu_info['available'], bool)
    assert isinstance(gpu_info['device_count'], int)
    assert isinstance(gpu_info['pytorch_available'], bool)
    assert isinstance(gpu_info['xgboost_gpu_support'], bool)

    print(f"✅ GPU Info: {gpu_info}")


def test_get_xgboost_gpu_params():
    """Test XGBoost GPU params"""
    params = get_xgboost_gpu_params()

    # Check required keys
    assert 'tree_method' in params
    assert 'device' in params

    # Check values
    assert params['tree_method'] in ['hist', 'auto']
    assert params['device'] in ['cuda', 'cpu']

    print(f"✅ XGBoost Params: {params}")


def test_get_pytorch_device():
    """Test PyTorch device detection"""
    device = get_pytorch_device()

    assert device in ['cuda', 'cpu']
    print(f"✅ PyTorch Device: {device}")


if __name__ == "__main__":
    print("🧪 Testing GPU utilities...")
    print("="*60)

    test_detect_gpu()
    test_get_xgboost_gpu_params()
    test_get_pytorch_device()

    print("="*60)
    print("✅ All tests passed!")