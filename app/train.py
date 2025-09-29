"""
訓練與校準模組 - 完整實作
包含 LogisticRegression、XGBoost、校準、交叉驗證與模型持久化
"""
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
import warnings
from datetime import datetime

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
    GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.isotonic import IsotonicRegression

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost 未安裝，將使用 RandomForest 作為替代")


class ExoplanetTrainer:
    """系外行星偵測模型訓練器"""

    def __init__(
        self,
        model_type: str = "logistic",
        calibration_method: str = "isotonic",
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        初始化訓練器

        Parameters:
        -----------
        model_type : str
            模型類型 ('logistic', 'xgboost', 'random_forest')
        calibration_method : str
            校準方法 ('isotonic', 'sigmoid')
        random_state : int
            隨機種子
        verbose : bool
            是否輸出訓練資訊
        """
        self.model_type = model_type
        self.calibration_method = calibration_method
        self.random_state = random_state
        self.verbose = verbose

        self.model = None
        self.calibrated_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_metrics = {}

    def create_base_model(self) -> Any:
        """建立基礎模型"""
        if self.model_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif self.model_type == "xgboost" and HAS_XGBOOST:
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='binary:logistic',
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:  # random_forest or fallback
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state,
                class_weight='balanced'
            )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        訓練模型並進行校準

        Parameters:
        -----------
        X_train : np.ndarray
            訓練特徵
        y_train : np.ndarray
            訓練標籤
        X_val : np.ndarray, optional
            驗證特徵
        y_val : np.ndarray, optional
            驗證標籤
        feature_names : List[str], optional
            特徵名稱
        cv_folds : int
            交叉驗證折數

        Returns:
        --------
        metrics : Dict[str, Any]
            訓練指標
        """
        if self.verbose:
            print(f"🚀 開始訓練 {self.model_type} 模型...")
            print(f"   訓練樣本數: {len(X_train)}")
            print(f"   特徵維度: {X_train.shape[1]}")
            print(f"   正樣本比例: {y_train.mean():.2%}")

        # 儲存特徵名稱
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

        # 特徵標準化
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 建立基礎模型
        base_model = self.create_base_model()

        # 交叉驗證評估
        if self.verbose:
            print(f"\n📊 執行 {cv_folds} 折交叉驗證...")

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(base_model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')

        if self.verbose:
            print(f"   CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # 訓練基礎模型
        base_model.fit(X_train_scaled, y_train)
        self.model = base_model

        # 機率校準
        if self.verbose:
            print(f"\n🎯 進行機率校準 (方法: {self.calibration_method})...")

        self.calibrated_model = CalibratedClassifierCV(
            base_model,
            method=self.calibration_method,
            cv=3
        )
        self.calibrated_model.fit(X_train_scaled, y_train)

        # 評估模型
        metrics = self._evaluate_model(X_train_scaled, y_train, "訓練集")

        # 如果提供驗證集
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_metrics = self._evaluate_model(X_val_scaled, y_val, "驗證集")
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

        self.training_metrics = metrics

        # 計算特徵重要性
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            if self.verbose:
                print("\n🏆 前 10 個重要特徵:")
                for idx, row in self.feature_importance.head(10).iterrows():
                    print(f"   {row['feature']:20} : {row['importance']:.4f}")

        return metrics

    def _evaluate_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str
    ) -> Dict[str, float]:
        """評估模型效能"""
        # 取得預測
        y_pred = self.calibrated_model.predict(X)
        y_proba = self.calibrated_model.predict_proba(X)[:, 1]

        # 計算指標
        roc_auc = roc_auc_score(y, y_proba)
        precision, recall, _ = precision_recall_curve(y, y_proba)
        pr_auc = auc(recall, precision)

        # 分類指標
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)

        if self.verbose:
            print(f"\n📈 {dataset_name}效能:")
            print(f"   ROC-AUC: {roc_auc:.4f}")
            print(f"   PR-AUC: {pr_auc:.4f}")
            print(f"   Precision: {prec:.4f}")
            print(f"   Recall: {rec:.4f}")
            print(f"   F1-Score: {f1:.4f}")

        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        }

    def predict(
        self,
        X: np.ndarray,
        return_proba: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        進行預測

        Parameters:
        -----------
        X : np.ndarray
            特徵矩陣
        return_proba : bool
            是否回傳機率

        Returns:
        --------
        predictions : np.ndarray or Tuple
            預測結果
        """
        if self.calibrated_model is None:
            raise ValueError("模型尚未訓練")

        X_scaled = self.scaler.transform(X)

        if return_proba:
            y_proba = self.calibrated_model.predict_proba(X_scaled)[:, 1]
            y_pred = (y_proba >= 0.5).astype(int)
            return y_pred, y_proba
        else:
            return self.calibrated_model.predict(X_scaled)

    def save(
        self,
        output_dir: str = "model",
        model_name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        儲存模型與相關檔案

        Parameters:
        -----------
        output_dir : str
            輸出目錄
        model_name : str, optional
            模型名稱

        Returns:
        --------
        paths : Dict[str, str]
            儲存的檔案路徑
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 建立檔案名稱
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{self.model_type}_{timestamp}"

        paths = {}

        # 儲存校準後的模型
        model_path = output_path / f"{model_name}_calibrated.joblib"
        joblib.dump(self.calibrated_model, model_path)
        paths['calibrated_model'] = str(model_path)

        # 儲存基礎模型
        base_model_path = output_path / f"{model_name}_base.joblib"
        joblib.dump(self.model, base_model_path)
        paths['base_model'] = str(base_model_path)

        # 儲存 scaler
        scaler_path = output_path / f"{model_name}_scaler.joblib"
        joblib.dump(self.scaler, scaler_path)
        paths['scaler'] = str(scaler_path)

        # 儲存特徵結構
        feature_schema = {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'model_type': self.model_type,
            'calibration_method': self.calibration_method,
            'training_metrics': self.training_metrics,
            'timestamp': datetime.now().isoformat()
        }

        schema_path = output_path / f"{model_name}_schema.json"
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(feature_schema, f, indent=2, ensure_ascii=False)
        paths['schema'] = str(schema_path)

        # 儲存特徵重要性
        if hasattr(self, 'feature_importance'):
            importance_path = output_path / f"{model_name}_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)
            paths['importance'] = str(importance_path)

        if self.verbose:
            print(f"\n💾 模型已儲存至 {output_dir}/")
            for key, path in paths.items():
                print(f"   {key}: {Path(path).name}")

        return paths

    @classmethod
    def load(
        cls,
        model_path: str,
        verbose: bool = True
    ) -> 'ExoplanetTrainer':
        """
        載入已儲存的模型

        Parameters:
        -----------
        model_path : str
            模型路徑 (可以是目錄或具體檔案)
        verbose : bool
            是否輸出資訊

        Returns:
        --------
        trainer : ExoplanetTrainer
            載入的訓練器實例
        """
        model_path = Path(model_path)

        # 如果是目錄，尋找最新的模型
        if model_path.is_dir():
            calibrated_files = list(model_path.glob("*_calibrated.joblib"))
            if not calibrated_files:
                raise FileNotFoundError(f"在 {model_path} 找不到校準模型")
            model_file = max(calibrated_files, key=lambda x: x.stat().st_mtime)
            base_name = str(model_file).replace("_calibrated.joblib", "")
        else:
            base_name = str(model_path).replace("_calibrated.joblib", "")
            model_file = model_path

        # 載入各個元件
        calibrated_model = joblib.load(f"{base_name}_calibrated.joblib")
        base_model = joblib.load(f"{base_name}_base.joblib")
        scaler = joblib.load(f"{base_name}_scaler.joblib")

        # 載入 schema
        with open(f"{base_name}_schema.json", 'r', encoding='utf-8') as f:
            schema = json.load(f)

        # 建立訓練器實例
        trainer = cls(
            model_type=schema['model_type'],
            calibration_method=schema['calibration_method'],
            verbose=verbose
        )

        trainer.calibrated_model = calibrated_model
        trainer.model = base_model
        trainer.scaler = scaler
        trainer.feature_names = schema['feature_names']
        trainer.training_metrics = schema.get('training_metrics', {})

        # 載入特徵重要性
        importance_file = Path(f"{base_name}_importance.csv")
        if importance_file.exists():
            trainer.feature_importance = pd.read_csv(importance_file)

        if verbose:
            print(f"✅ 成功載入模型")
            print(f"   模型類型: {trainer.model_type}")
            print(f"   校準方法: {trainer.calibration_method}")
            print(f"   特徵數量: {len(trainer.feature_names)}")

        return trainer


def hyperparameter_tuning(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = "xgboost",
    cv_folds: int = 3,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    超參數調優

    Parameters:
    -----------
    X_train : np.ndarray
        訓練特徵
    y_train : np.ndarray
        訓練標籤
    model_type : str
        模型類型
    cv_folds : int
        交叉驗證折數
    verbose : bool
        是否輸出資訊

    Returns:
    --------
    best_params : Dict[str, Any]
        最佳參數
    """
    if verbose:
        print(f"🔍 開始 {model_type} 超參數搜尋...")

    # 定義參數網格
    if model_type == "logistic":
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
        base_model = LogisticRegression(max_iter=1000, random_state=42)

    elif model_type == "xgboost" and HAS_XGBOOST:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 10],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0]
        }
        base_model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42
        )

    else:  # random_forest
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        base_model = RandomForestClassifier(random_state=42)

    # 執行網格搜尋
    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=cv_folds,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1 if verbose else 0
    )

    grid_search.fit(X_train, y_train)

    if verbose:
        print(f"\n✨ 最佳參數:")
        for param, value in grid_search.best_params_.items():
            print(f"   {param}: {value}")
        print(f"\n   最佳 CV 分數: {grid_search.best_score_:.4f}")

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'grid_search': grid_search
    }


# 保留原始簡單介面的相容性
def train_and_save(
    X: np.ndarray,
    y: np.ndarray,
    out_dir: str = "model"
) -> None:
    """
    簡單的訓練與儲存介面（向後相容）
    """
    trainer = ExoplanetTrainer(model_type="logistic", verbose=True)
    trainer.train(X, y)
    trainer.save(out_dir, model_name="ranker")


if __name__ == "__main__":
    # 測試範例
    print("🧪 訓練模組測試")

    # 產生測試資料
    np.random.seed(42)
    X_test = np.random.randn(1000, 14)
    y_test = np.random.randint(0, 2, 1000)

    # 測試訓練流程
    trainer = ExoplanetTrainer(model_type="logistic")
    metrics = trainer.train(X_test, y_test, cv_folds=3)

    # 儲存模型
    paths = trainer.save("test_model")

    # 載入模型測試
    loaded_trainer = ExoplanetTrainer.load("test_model")
    predictions, probas = loaded_trainer.predict(X_test[:10])

    print(f"\n✅ 測試完成！預測機率: {probas}")