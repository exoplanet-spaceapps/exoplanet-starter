"""
Reproducibility Utilities for Exoplanet Detection Pipeline
确保所有随机操作具有可重现性
"""
import random
import numpy as np
import os


def set_random_seeds(seed: int = 42) -> None:
    """
    设置所有随机种子以确保可重现性

    Args:
        seed: 随机种子值，默认为 42

    Sets seeds for:
        - Python random module
        - NumPy
        - PyTorch (如果已安装)
        - CUDA (如果可用)
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (如果已安装)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # 确保 CUDA 操作的确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # 设置环境变量以确保 Python 哈希种子也固定
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"🎲 Random seeds set to {seed} for reproducibility")


def get_random_state():
    """
    返回当前所有随机状态的快照

    Returns:
        dict: 包含所有随机状态的字典
    """
    state = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
    }

    try:
        import torch
        state['torch_random'] = torch.get_rng_state()
        if torch.cuda.is_available():
            state['torch_cuda_random'] = torch.cuda.get_rng_state_all()
    except ImportError:
        pass

    return state


def restore_random_state(state: dict) -> None:
    """
    恢复之前保存的随机状态

    Args:
        state: 由 get_random_state() 返回的状态字典
    """
    random.setstate(state['python_random'])
    np.random.set_state(state['numpy_random'])

    try:
        import torch
        if 'torch_random' in state:
            torch.set_rng_state(state['torch_random'])
        if 'torch_cuda_random' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['torch_cuda_random'])
    except ImportError:
        pass

    print("🔄 Random state restored")