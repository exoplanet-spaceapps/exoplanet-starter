"""
Sklearn Pipeline for Exoplanet Detection (Phase 3)
完整的机器学习管道，包含前处理和模型训练
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from typing import List, Dict, Any, Optional


def create_exoplanet_pipeline(
    numerical_features: List[str],
    xgb_params: Optional[Dict[str, Any]] = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42
) -> Pipeline:
    """
    Create complete exoplanet detection pipeline with preprocessing and XGBoost

    Pipeline steps:
    1. SimpleImputer: Fill missing values with median
    2. RobustScaler: Scale features (robust to outliers)
    3. XGBClassifier: Train gradient boosting model with GPU support

    Args:
        numerical_features: List of numerical feature column names
        xgb_params: Dict of XGBoost parameters (e.g., {'device': 'cuda', 'tree_method': 'hist'})
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Boosting learning rate
        random_state: Random seed for reproducibility

    Returns:
        sklearn.pipeline.Pipeline: Complete pipeline ready for training

    Example:
        >>> from utils import get_xgboost_gpu_params
        >>> gpu_params = get_xgboost_gpu_params()
        >>> pipeline = create_exoplanet_pipeline(
        ...     numerical_features=['bls_period', 'bls_depth_ppm'],
        ...     xgb_params=gpu_params,
        ...     random_state=42
        ... )
        >>> pipeline.fit(X_train, y_train)
        >>> predictions = pipeline.predict_proba(X_test)
    """
    if xgb_params is None:
        xgb_params = {}

    # Numerical preprocessing pipeline
    # - SimpleImputer: Handles missing values with median (robust to outliers)
    # - RobustScaler: Scales based on median and IQR (better than StandardScaler for outliers)
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    # Column transformer to apply transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features)
        ],
        remainder='drop'  # Drop non-numerical columns
    )

    # Complete pipeline with XGBoost classifier
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            **xgb_params,  # GPU params (device='cuda' for XGBoost 2.x)
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,  # Reproducibility
            eval_metric='aucpr'  # PR-AUC (better for imbalanced data)
            # early_stopping_rounds removed - not compatible with pipeline without eval_set
        ))
    ])

    return pipeline