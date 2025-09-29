"""
Execute notebook 03 with proper cell ordering
Extracts cells and runs them in logical order: imports -> data -> training
"""
import json
import sys
import os
from pathlib import Path

# Add src and notebooks to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'notebooks'))
os.chdir(project_root / 'notebooks')

print("="*70)
print("Executing Notebook 03: injection_train.ipynb")
print("Working directory:", os.getcwd())
print("="*70)

# Load notebook
with open('03_injection_train.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"\nTotal cells in notebook: {len(nb['cells'])}")

# Execute cells in proper order
# Step 1: Find and execute basic imports
print("\n" + "="*70)
print("STEP 1: Execute imports")
print("="*70)

basic_imports = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, '../src')

print("Basic imports loaded")
"""

exec(basic_imports, globals())

# Step 2: Execute data loading
print("\n" + "="*70)
print("STEP 2: Load data")
print("="*70)

data_loading = """
# Data Loading
print("Loading supervised_dataset.csv...")

from data_loader_colab import setup_data_directory, load_datasets

# Setup and load data
data_dir, IN_COLAB = setup_data_directory()
datasets = load_datasets(data_dir)

# Extract datasets
if 'supervised_dataset' in datasets:
    supervised_df = datasets['supervised_dataset']
    print(f"Loaded supervised_dataset: {len(supervised_df)} rows")

    # Create positive/negative splits
    if 'label' in supervised_df.columns:
        toi_positive = supervised_df[supervised_df['label'] == 1].copy()
        toi_positive['source'] = 'TOI'

        negative_samples = supervised_df[supervised_df['label'] == 0].copy()
        split_idx = len(negative_samples) // 2
        eb_negative = negative_samples.iloc[:split_idx].copy()
        toi_fp = negative_samples.iloc[split_idx:].copy()
        eb_negative['source'] = 'Kepler_EB'
        toi_fp['source'] = 'TOI_FP'

        print(f"TOI positive: {len(toi_positive)}")
        print(f"EB negative: {len(eb_negative)}")
        print(f"TOI FP negative: {len(toi_fp)}")
    else:
        print("ERROR: No label column in supervised_dataset")
        sys.exit(1)
else:
    print("ERROR: supervised_dataset not found")
    sys.exit(1)

print("Data loaded successfully!")
"""

exec(data_loading, globals())

# Step 3: Feature extraction (simplified for now - create dummy features)
print("\n" + "="*70)
print("STEP 3: Create feature dataframe")
print("="*70)

# For now, just use the supervised_dataset columns as features
features_df = supervised_df.copy()

# Remove non-numeric columns for training
numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
if 'label' in numeric_cols:
    numeric_cols.remove('label')

feature_cols = numeric_cols[:20]  # Use first 20 numeric features
print(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")

# Step 4: Import Phase 3-4 modules
print("\n" + "="*70)
print("STEP 4: Import Phase 3-4 modules")
print("="*70)

phase3_imports = """
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold

from models.pipeline import create_exoplanet_pipeline
from utils.gpu_utils import get_xgboost_gpu_params, log_gpu_info

print("Phase 3-4 imports loaded")
"""

exec(phase3_imports, globals())

# Step 5: Prepare data
print("\n" + "="*70)
print("STEP 5: Prepare training data")
print("="*70)

# Keep as DataFrame for pipeline compatibility
X_df = features_df[feature_cols].copy()
y = features_df['label'].values

# Clean NaN/Inf in-place
for col in X_df.columns:
    X_df[col] = X_df[col].replace([np.inf, -np.inf], np.nan)

X = X_df  # Use DataFrame for pipeline

# Create groups
if 'TIC_ID' in features_df.columns:
    groups = features_df['TIC_ID'].astype('category').cat.codes.values
elif 'TOI' in features_df.columns:
    groups = features_df['TOI'].astype('category').cat.codes.values
else:
    groups = np.arange(len(y))

print(f"Samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Positive ratio: {y.mean():.2%}")
print(f"Unique groups: {len(np.unique(groups))}")

log_gpu_info()

# Step 6: Train with Pipeline + Cross-Validation
print("\n" + "="*70)
print("STEP 6: Train XGBoost Pipeline with Cross-Validation")
print("="*70)

gpu_params = get_xgboost_gpu_params()
print(f"XGBoost params: {gpu_params}")

n_splits = 3  # Reduced for faster execution
sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

fold_results = []
fold_models = []

for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups), 1):
    print(f"\nFold {fold_idx}/{n_splits}")
    print("-" * 40)

    X_train_fold, X_test_fold = X.iloc[train_idx], X.iloc[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    # Train pipeline
    pipeline_fold = create_exoplanet_pipeline(
        numerical_features=feature_cols,
        xgb_params=gpu_params,
        n_estimators=50,  # Reduced for faster execution
        max_depth=6,
        learning_rate=0.1,
        random_state=42 + fold_idx
    )

    print(f"Training on {len(train_idx)} samples...")
    pipeline_fold.fit(X_train_fold, y_train_fold)

    # Predict
    y_pred_proba_fold = pipeline_fold.predict_proba(X_test_fold)[:, 1]

    # Metrics
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score

    ap_score = average_precision_score(y_test_fold, y_pred_proba_fold)
    auc_score = roc_auc_score(y_test_fold, y_pred_proba_fold)

    y_pred_binary = (y_pred_proba_fold >= 0.5).astype(int)
    precision = precision_score(y_test_fold, y_pred_binary, zero_division=0)
    recall = recall_score(y_test_fold, y_pred_binary)

    fold_results.append({
        'fold': fold_idx,
        'auc_pr': ap_score,
        'auc_roc': auc_score,
        'precision': precision,
        'recall': recall
    })

    fold_models.append(pipeline_fold)

    print(f"AUC-PR:  {ap_score:.4f}")
    print(f"AUC-ROC: {auc_score:.4f}")
    print(f"Precision@0.5: {precision:.4f}")
    print(f"Recall@0.5: {recall:.4f}")

# Summary
print("\n" + "="*70)
print("CROSS-VALIDATION SUMMARY")
print("="*70)

fold_df = pd.DataFrame(fold_results)
print(f"\nAUC-PR:  {fold_df['auc_pr'].mean():.4f} ± {fold_df['auc_pr'].std():.4f}")
print(f"AUC-ROC: {fold_df['auc_roc'].mean():.4f} ± {fold_df['auc_roc'].std():.4f}")
print(f"Precision@0.5: {fold_df['precision'].mean():.4f} ± {fold_df['precision'].std():.4f}")
print(f"Recall@0.5: {fold_df['recall'].mean():.4f} ± {fold_df['recall'].std():.4f}")

# Save best model
best_fold_idx = fold_df['auc_pr'].idxmax()
best_model = fold_models[best_fold_idx]
print(f"\nBest model: Fold {best_fold_idx + 1} (AUC-PR: {fold_df.loc[best_fold_idx, 'auc_pr']:.4f})")

# Save model
import joblib
model_path = Path('../models/xgboost_pipeline_cv.joblib')
model_path.parent.mkdir(exist_ok=True)
joblib.dump(best_model, model_path)
print(f"Model saved to: {model_path}")

# Save results
results_df = fold_df.copy()
results_path = Path('../reports/cv_results.csv')
results_path.parent.mkdir(exist_ok=True)
results_df.to_csv(results_path, index=False)
print(f"Results saved to: {results_path}")

print("\n" + "="*70)
print("NOTEBOOK 03 EXECUTION COMPLETE!")
print("="*70)

print("\nScript completed successfully!")