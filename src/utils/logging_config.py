"""
Logging Configuration for Exoplanet Detection Pipeline
统一的日志配置系统，适用于 Colab 和本地环境
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "exoplanet",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    verbose: bool = True
) -> logging.Logger:
    """
    设置统一的日志配置

    Args:
        name: Logger 名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径，如果为 None 则只输出到控制台
        verbose: 是否显示详细信息（包括时间戳）

    Returns:
        logging.Logger: 配置好的 logger 对象

    Example:
        >>> logger = setup_logger("my_notebook", level=logging.DEBUG)
        >>> logger.info("Starting analysis...")
        >>> logger.warning("Low SNR detected")
    """
    # 创建 logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清除现有的 handlers（避免重复添加）
    logger.handlers.clear()

    # 设置日志格式
    if verbose:
        # 详细格式：包含时间、级别、名称、消息
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        # 简洁格式：只有级别和消息
        formatter = logging.Formatter('%(levelname)s - %(message)s')

    # 控制台 handler（始终添加）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件 handler（如果指定了 log_file）
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"📝 Logging to file: {log_file}")

    # 防止日志传播到父 logger（避免重复输出）
    logger.propagate = False

    return logger


def get_log_file_path(notebook_name: str, results_dir: Path = Path("results")) -> Path:
    """
    为指定的 notebook 生成日志文件路径

    Args:
        notebook_name: Notebook 名称（例如 "02_bls_baseline"）
        results_dir: 结果目录路径

    Returns:
        Path: 日志文件路径

    Example:
        >>> log_path = get_log_file_path("02_bls_baseline")
        >>> logger = setup_logger("02_bls", log_file=log_path)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{notebook_name}_{timestamp}.log"
    log_dir = results_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / log_filename


def log_system_info(logger: logging.Logger) -> None:
    """
    记录系统信息（Python 版本、环境、GPU 可用性等）

    Args:
        logger: Logger 对象
    """
    import platform
    import sys

    logger.info("=" * 60)
    logger.info("System Information")
    logger.info("=" * 60)
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Processor: {platform.processor()}")

    # 检查 GPU 可用性
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        logger.info("PyTorch not installed")

    # 检查其他关键库
    try:
        import numpy as np
        logger.info(f"NumPy version: {np.__version__}")
    except ImportError:
        pass

    try:
        import pandas as pd
        logger.info(f"Pandas version: {pd.__version__}")
    except ImportError:
        pass

    try:
        import sklearn
        logger.info(f"Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        pass

    try:
        import xgboost as xgb
        logger.info(f"XGBoost version: {xgb.__version__}")
    except ImportError:
        pass

    logger.info("=" * 60)


def log_data_info(logger: logging.Logger, data_dict: dict) -> None:
    """
    记录数据集信息

    Args:
        logger: Logger 对象
        data_dict: 数据集字典 (名称 -> DataFrame)

    Example:
        >>> logger = setup_logger()
        >>> datasets = {'train': train_df, 'test': test_df}
        >>> log_data_info(logger, datasets)
    """
    logger.info("=" * 60)
    logger.info("Dataset Information")
    logger.info("=" * 60)

    for name, df in data_dict.items():
        logger.info(f"{name}:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # 如果有标签列，显示类别分布
        if 'label' in df.columns:
            label_counts = df['label'].value_counts()
            logger.info(f"  Label distribution: {dict(label_counts)}")

    logger.info("=" * 60)