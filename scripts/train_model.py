#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train exoplanet detection model using XGBoost"""

import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib

warnings.filterwarnings('ignore')

# 配置
PROJECT_ROOT = Path(__file__).parent.parent
FEATURES_PATH = PROJECT_ROOT / 'data' / 'balanced_features.csv'
MODEL_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("Exoplanet Detection Model Training")
print("="*70)

# 載入特徵
print("\n[1/6] Loading features...")
if not FEATURES_PATH.exists():
    print(f"❌ Features file not found: {FEATURES_PATH}")
    print("Run: python scripts/test_features.py")
    exit(1)

df = pd.read_csv(FEATURES_PATH)
print(f"  Total samples: {len(df)}")

# 篩選成功提取的樣本
successful = df[df['status'] == 'success'].copy()
print(f"  Successful extractions: {len(successful)}")

if len(successful) < 100:
    print("❌ Not enough successful samples for training (minimum 100)")
    exit(1)

# 準備特徵與標籤
print("\n[2/6] Preparing features...")
feature_cols = [
    'flux_mean', 'flux_std', 'flux_median', 'flux_mad', 'flux_skew', 'flux_kurt',
    'bls_period', 'bls_duration', 'bls_depth', 'bls_power', 'bls_snr'
]

X = successful[feature_cols].values
y = successful['label'].values

print(f"  Features: {X.shape}")
print(f"  Positive samples: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"  Negative samples: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")

# 分割資料
print("\n[3/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"  Train: {X_train.shape[0]} samples")
print(f"  Test:  {X_test.shape[0]} samples")

# 訓練模型（先試 XGBoost，fallback 到 RandomForest）
print("\n[4/6] Training model...")
try:
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model_type = 'XGBoost'
    print(f"  Using {model_type}")

except ImportError:
    print("  XGBoost not found, using RandomForest instead")
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model_type = 'RandomForest'

model.fit(X_train, y_train)
print("  ✅ Training complete")

# 交叉驗證
print("\n[5/6] Cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
print(f"  CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 評估
print("\n[6/6] Evaluation...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

metrics = {
    'model_type': model_type,
    'n_samples': len(successful),
    'n_features': len(feature_cols),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred)),
    'f1_score': float(f1_score(y_test, y_pred)),
    'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
    'cv_roc_auc_mean': float(cv_scores.mean()),
    'cv_roc_auc_std': float(cv_scores.std()),
    'timestamp': datetime.now().isoformat()
}

print(f"\nTest Set Performance:")
print(f"  Accuracy:  {metrics['accuracy']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall:    {metrics['recall']:.4f}")
print(f"  F1 Score:  {metrics['f1_score']:.4f}")
print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
print(f"  FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")

# Classification Report
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Planet', 'Exoplanet']))

# Feature Importance
if hasattr(model, 'feature_importances_'):
    importance = model.feature_importances_
    feature_importance = sorted(
        zip(feature_cols, importance),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\nFeature Importance:")
    for feat, imp in feature_importance[:5]:
        print(f"  {feat:20s}: {imp:.4f}")

    metrics['feature_importance'] = {
        feat: float(imp) for feat, imp in feature_importance
    }

# 保存模型
model_path = MODEL_DIR / f'exoplanet_model_{model_type.lower()}.pkl'
joblib.dump(model, model_path)
print(f"\n✅ Model saved: {model_path}")

# 保存評估結果
results_path = RESULTS_DIR / 'training_results.json'
with open(results_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"✅ Results saved: {results_path}")

# 保存特徵名稱
features_meta_path = MODEL_DIR / 'feature_names.json'
with open(features_meta_path, 'w') as f:
    json.dump({'features': feature_cols}, f, indent=2)
print(f"✅ Feature metadata saved: {features_meta_path}")

print("="*70)

if metrics['roc_auc'] >= 0.80:
    print("🎉 Model training successful! (ROC-AUC ≥ 0.80)")
    print("\nNext steps:")
    print("  1. Review results: cat results/training_results.json")
    print("  2. Test inference: python scripts/predict.py")
    print("  3. Deploy to production")
else:
    print(f"⚠️ Model performance below target (ROC-AUC = {metrics['roc_auc']:.4f})")
    print("\nSuggestions:")
    print("  1. Collect more training data")
    print("  2. Feature engineering (add more BLS parameters)")
    print("  3. Hyperparameter tuning")

print("="*70)
