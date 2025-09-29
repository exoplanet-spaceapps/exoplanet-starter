"""
訓練與校準（示意接口，Notebook 內有完整版本）
"""
from typing import List, Dict
import json, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

def train_and_save(X, y, out_dir="model"):
    base = LogisticRegression(max_iter=200)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X, y)
    import os
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(clf, f"{out_dir}/ranker.joblib")
    with open(f"{out_dir}/feature_schema.json", "w") as f:
        json.dump({"features": list(range(X.shape[1]))}, f)
