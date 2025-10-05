# 📊 數據切分策略說明

**Exoplanet Detection - Training/Validation/Testing Data Split Strategy**

本文檔詳細說明本專案中機器學習訓練時的數據來源、切分比例與使用策略。

---

## 📋 目錄

1. [數據來源](#數據來源)
2. [數據切分流程](#數據切分流程)
3. [各模型的切分策略](#各模型的切分策略)
4. [關鍵設計決策](#關鍵設計決策)

---

## 數據來源

### 原始數據集
```
MAST Archive (TESS)
    ↓
supervised_dataset.csv (11,979 筆)
├─ True (有系外行星): 5,944 筆
└─ False (無系外行星): 6,035 筆
```

**來源**: NASA TESS 任務的光變曲線數據，通過 MAST Archive 下載

---

## 數據切分流程

### 完整流程圖

```
┌─────────────────────────────────────────────────────┐
│ MAST Archive (TESS)                                 │
│ 原始 Lightcurve 數據                                │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│ supervised_dataset.csv (11,979 筆)                  │
│ ├─ True (有行星): 5,944 筆                          │
│ └─ False (無行星): 6,035 筆                         │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼ [extract_balanced_features.py]
                   │ (隨機抽樣, random_state=42)
┌─────────────────────────────────────────────────────┐
│ balanced_features.csv (1,000 筆)                    │
│ ├─ True: 500 筆                                     │
│ └─ False: 500 筆                                    │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼ [train_test_split]
                   │ test_size=0.2, stratify=y
        ┌──────────┴──────────┐
        │                     │
        ▼                     ▼
┌──────────────┐    ┌──────────────┐
│ 訓練集 (80%) │    │ 測試集 (20%) │
│ 800 筆       │    │ 200 筆       │
│ ├─ True: 400│    │ ├─ True: 100│
│ └─ False:400│    │ └─ False:100│
└──────┬───────┘    └──────────────┘
       │                    ▲
       │                    │
       │                    │ 最終評估
       ▼                    │
  [兩種策略]                 │
       │                    │
  ┌────┴────┐               │
  │         │               │
  ▼         ▼               │
┌────┐  ┌────────┐          │
│直接│  │5-Fold  │          │
│訓練│  │Cross   │          │
│    │  │Valid.  │          │
└─┬──┘  └───┬────┘          │
  │         │               │
  │    ┌────┴─────┐         │
  │    ▼          ▼         │
  │  Fold1-5   選最佳       │
  │  (內部)    超參數        │
  │            │            │
  └────────────┴────────────┘
               │
               ▼
          最終模型
```

---

## 各模型的切分策略

### 1️⃣ XGBoost Baseline (`train_model_local.py`)

#### 數據切分
```python
# Step 1: 平衡抽樣
balanced_features.csv: 1,000 筆
├─ True: 500 筆 (從 5,944 筆中隨機抽)
└─ False: 500 筆 (從 6,035 筆中隨機抽)

# Step 2: Train-Test Split (80-20)
train_test_split(test_size=0.2, random_state=42, stratify=y)

訓練集 (X_train, y_train): 800 筆
├─ True: 400 筆 (50%)
└─ False: 400 筆 (50%)

測試集 (X_test, y_test): 200 筆
├─ True: 100 筆 (50%)
└─ False: 100 筆 (50%)
```

#### 訓練流程
- **訓練**: 直接在 800 筆訓練集上訓練 XGBoost 模型
- **評估**: 在 200 筆測試集上評估最終效能
- **無 Validation Set**: 沒有使用獨立的驗證集或交叉驗證

#### 程式碼片段
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

# 最終評估
y_pred = model.predict(X_test)
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
```

---

### 2️⃣ XGBoost Optuna (`train_xgboost_optuna.py`)

#### 數據切分
相同的 Train-Test Split (800 訓練 + 200 測試)

#### 額外使用 5-Fold Cross-Validation
```python
cross_val_score(model, X_train, y_train, cv=5)

每個 Fold 的切分 (在 800 筆訓練集內):
├─ Training Fold: 640 筆 (80% of 800)
└─ Validation Fold: 160 筆 (20% of 800)

重複 5 次，每次不同的 Fold 作為 Validation
```

#### 訓練流程

**階段 1: Optuna 超參數搜索**
1. Optuna 生成候選超參數組合
2. 對每組超參數：
   - 在訓練集 (800 筆) 上做 **5-Fold Cross-Validation**
   - 每個 Fold 分為 Training (640) 和 Validation (160)
   - 計算 5 個 Fold 的平均 ROC-AUC
3. 選擇平均 ROC-AUC 最高的超參數組合

**階段 2: 最終模型訓練**
1. 使用最佳超參數
2. 在整個訓練集 (800 筆) 上訓練
3. 在測試集 (200 筆) 上評估

#### Validation Set 來源
- **內部 Validation**: 5-Fold CV 中的每個 Validation Fold (160 筆)
- 這些 Validation Fold 是從訓練集 (800 筆) 中**動態切分**的
- **不是**獨立的第三個數據集，而是從訓練集中輪流抽取

#### 程式碼片段
```python
def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        # ... 其他超參數
    }

    model = xgb.XGBClassifier(**params)

    # 5-Fold Cross-Validation (只在訓練集上)
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring='roc_auc', n_jobs=-1
    )

    return cv_scores.mean()  # 返回平均 ROC-AUC

# Optuna 搜索
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

# 使用最佳超參數訓練最終模型
best_model = xgb.XGBClassifier(**study.best_params)
best_model.fit(X_train, y_train)
```

---

### 3️⃣ Genesis CNN (`train_genesis_cnn.py`)

#### 數據切分
相同的 Train-Test Split (800 訓練 + 200 測試)

#### PyTorch DataLoader
```python
訓練集: 800 筆 → DataLoader (batch_size=32)
測試集: 200 筆 → DataLoader (batch_size=32)

訓練過程：
├─ 每個 Epoch: 在訓練集 (800 筆) 上訓練
└─ 每個 Epoch 結束: 在測試集 (200 筆) 上驗證
```

#### Validation Set 策略
- **直接使用測試集作為 Validation**
- 每個 Epoch 訓練完後，在測試集上計算 ROC-AUC
- 儲存 Validation ROC-AUC 最高的模型

#### 訓練流程
```python
for epoch in range(EPOCHS):
    # Training
    model.train()
    for batch_flux, batch_labels in train_loader:
        # ... 訓練步驟

    # Validation (使用測試集)
    model.eval()
    for batch_flux, batch_labels in test_loader:
        # ... 驗證步驟

    val_auc = roc_auc_score(all_labels, all_preds)

    # 儲存最佳模型
    if val_auc > best_roc_auc:
        best_roc_auc = val_auc
        torch.save(model.state_dict(), 'genesis_cnn_best.pth')
```

#### ⚠️ 注意事項
- 測試集同時作為 Validation 和最終評估
- 這不是理想做法（會有數據洩漏風險）
- 理想情況應該有獨立的 Train/Validation/Test 三個集合
- 在小數據集 (1,000 筆) 的情況下，這是常見的權衡

---

### 4️⃣ Advanced XGBoost (`train_advanced_model.py`)

#### 特徵數量
- 21 個進階特徵（基礎 11 + 時序 4 + 頻域 3 + 小波 3）

#### 數據切分
與 XGBoost Baseline 完全相同：
- 訓練集: 800 筆
- 測試集: 200 筆
- 無額外 Validation Set

#### 超參數
使用 Optuna 調優後的最佳超參數（從 `train_xgboost_optuna.py` 得到）

---

## 數據切分總結表

| 模型 | 訓練集 | Validation | 測試集 | CV Folds | 特徵數 |
|------|--------|-----------|--------|----------|--------|
| **XGBoost Baseline** | 800 筆 | ❌ 無 | 200 筆 | ❌ | 11 |
| **XGBoost Optuna** | 800 筆 | ✅ 5-Fold (內部) | 200 筆 | ✅ 5 | 11 |
| **Genesis CNN** | 800 筆 | 200 筆 (=測試集) | 200 筆 | ❌ | Raw data |
| **Advanced XGBoost** | 800 筆 | ❌ 無 | 200 筆 | ❌ | 21 |

---

## 關鍵設計決策

### 1. 平衡抽樣 (Balanced Sampling)

**為什麼需要？**
```
原始數據: 5,944 True vs 6,035 False (接近平衡但不完全)
抽樣後:   500 True vs 500 False (完全平衡)
```

**好處:**
- 避免類別不平衡問題
- 模型不會偏向多數類別
- 評估指標更有意義

**實作:**
```python
balanced_df = pd.concat([
    df_true.sample(n=500, random_state=42),   # 500 筆 True
    df_false.sample(n=500, random_state=42)   # 500 筆 False
])
```

---

### 2. Stratified Split (分層切分)

**作用:**
- 確保訓練集和測試集都保持 50%-50% 的類別比例
- 避免切分後類別不平衡

**實作:**
```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

**結果:**
```
訓練集: 400 True (50%) + 400 False (50%) = 800 筆
測試集: 100 True (50%) + 100 False (50%) = 200 筆
```

---

### 3. 固定 Random State

**目的:**
- 確保每次執行時數據切分結果一致
- 方便結果重現與比較

**實作:**
```python
random_state=42  # 所有隨機操作都使用相同的 seed
```

---

### 4. 80-20 切分比例

**選擇理由:**
- **80% 訓練**: 足夠的數據讓模型學習模式
- **20% 測試**: 足夠的數據評估泛化能力
- **標準做法**: ML 領域常見的切分比例

**在小數據集 (1,000 筆) 的權衡:**
- 訓練集 800 筆可能偏少（理想是數千筆）
- 測試集 200 筆足夠評估但統計誤差較大
- 未來可以使用全部 11,979 筆數據提升效能

---

### 5. Cross-Validation 策略

**僅在 Optuna 超參數調優時使用:**

**為什麼只用在 Optuna？**
- 超參數調優需要可靠的驗證分數
- 單次 Train-Val Split 可能有偏差
- 5-Fold CV 提供更穩健的評估

**為什麼其他模型不用 CV？**
- 訓練時間考量（CV 會增加 5 倍時間）
- 小數據集上 CV 的效益有限
- 主要關注最終測試集效能

---

### 6. Validation Set 的三種方式

#### 方式 1: 無獨立 Validation (Baseline & Advanced)
```
只有 Train 和 Test，直接在 Test 上評估
優點: 簡單直接
缺點: 無法提前調整模型
```

#### 方式 2: K-Fold CV Validation (Optuna)
```
在訓練集內做交叉驗證
優點: 充分利用訓練數據，評估穩健
缺點: 訓練時間長
```

#### 方式 3: 使用 Test 作為 Validation (Genesis CNN)
```
訓練時用 Test 監控，保存最佳模型
優點: 實作簡單，適合早停
缺點: 數據洩漏風險，測試集效能可能過於樂觀
```

---

## 數據流程總結

### 完整 Pipeline

```
1. 原始數據 (11,979 筆)
   ↓
2. 平衡抽樣 (1,000 筆, random_state=42)
   ↓
3. 特徵提取 (balanced_features.csv)
   ↓
4. Train-Test Split (80-20, stratified)
   ├─ 訓練集 800 筆
   └─ 測試集 200 筆
   ↓
5. 模型訓練
   ├─ Baseline: 直接訓練
   ├─ Optuna: 5-Fold CV → 最佳超參數 → 訓練
   ├─ Genesis CNN: 訓練 + Test Validation
   └─ Advanced: 使用 Optuna 超參數訓練
   ↓
6. 測試集評估 (200 筆)
   ├─ ROC-AUC
   ├─ Accuracy, Precision, Recall, F1
   └─ Confusion Matrix
```

---

## 最佳實踐建議

### 當前設置的優點
✅ 類別平衡 (50-50)
✅ Stratified split 保持比例
✅ Random state 固定可重現
✅ Optuna 使用 CV 避免過擬合

### 未來改進方向

#### 1. 使用更多數據
```
當前: 1,000 筆
未來: 11,979 筆全部數據
預期效益: ROC-AUC 可能提升 5-10%
```

#### 2. 三分切割 (Train/Val/Test)
```
訓練集: 70% (8,385 筆)
驗證集: 15% (1,797 筆)
測試集: 15% (1,797 筆)
```

#### 3. K-Fold 用於最終評估
```
對所有模型使用 5-Fold CV
報告平均效能和標準差
提供更可靠的效能估計
```

---

## 參考資料

### 相關檔案
- 數據抽樣: `scripts/extract_balanced_features.py`
- 訓練腳本:
  - `scripts/train_model_local.py` (Baseline)
  - `scripts/train_xgboost_optuna.py` (Optuna)
  - `scripts/train_genesis_cnn.py` (CNN)
  - `scripts/train_advanced_model.py` (Advanced)
- 訓練指南: `docs/ML_TRAINING_GUIDE.md`

### 數據來源
- MAST Archive: https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html
- TESS Mission: https://tess.mit.edu/

---

**最後更新**: 2025-10-05
**版本**: 1.0
**作者**: Exoplanet Detection Team
