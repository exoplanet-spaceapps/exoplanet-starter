# 🧪 快速测试指南：两步走方案（测试 → 全量）

## 📋 测试阶段流程（15-20 分钟）

### ✅ Step 1: 配置测试模式

打开 `notebooks/02a_download_lightcurves.ipynb`

**修改 Cell 4 配置**：
```python
CONFIG = {
    'max_workers': 4,        # 测试阶段保持 4 即可
    'max_retries': 3,
    'timeout': 60,
    'batch_size': 100,
    'save_interval': 20,     # 改为 20（更频繁保存）
    'test_mode': True,       # ⚠️ 重点：改为 True
}
```

**检查点**：
```python
# Cell 4 执行后应该看到：
⚠️ TEST MODE: Only processing 100 samples
✅ Dataset loaded: 100 samples
   Positive: 70
   Negative: 30
```

---

### ✅ Step 2: 执行下载测试

在 Jupyter Notebook 中：
1. 点击 **Cell → Run All**
2. 观察进度条（应显示 100/100）
3. 预计时间：**15-20 分钟**

**实时监控**：
```bash
# 在另一个终端查看进度
watch -n 30 'ls data/lightcurves/*.pkl | wc -l'
```

**成功标志**（Cell 6 输出）：
```
🎉 Download complete!
   Total time: 0.25 hours (15 分钟)
   Average: 9.0 sec/sample

📊 Final Statistics:
   success: 85-95 (正常范围)
   failed: 5-15 (部分样本可能无数据)
   Success rate: 85-95%
```

---

### ✅ Step 3: 验证下载数据

**查看 Cell 7 输出**：
```
✅ SAMPLE_000012_TIC88863718.pkl
   TIC ID: 88863718
   Sectors: 3 ([13, 26, 40])
   Data points: 18,315
   Time span: 27.4 days

📦 Storage:
   Total files: 87
   Total size: 0.35 GB
   Average size: 4.1 MB/file
```

**手动检查**（可选）：
```python
import joblib
from pathlib import Path

# 随机读取一个文件
test_file = list(Path('data/lightcurves').glob('*.pkl'))[0]
data = joblib.load(test_file)

print(f"Sample ID: {data['sample_id']}")
print(f"TIC ID: {data['tic_id']}")
print(f"Sectors: {data['n_sectors']}")
print(f"Light curves: {len(data['lc_collection'])}")

# 查看第一个光曲线
lc = data['lc_collection'][0]
print(f"Time points: {len(lc.time)}")
print(f"Time range: {lc.time[0]} - {lc.time[-1]}")
```

---

### ✅ Step 4: 测试特征提取

打开 `notebooks/02b_extract_features.ipynb`

**配置已经适用于测试**（无需修改）：
```python
CONFIG = {
    'max_workers': 4,
    'bls_periods': 2000,     # 测试用标准配置
    'period_max': 15.0,
}
```

**执行**：
1. 点击 **Cell → Run All**
2. 预计时间：**2-3 分钟**（100个样本）

**成功标志**（Cell 6 输出）：
```
✅ Feature extraction complete
   Total features: 87
   Feature columns: 14
   Features: ['flux_mean', 'flux_std', 'bls_power', ...]
```

---

### ✅ Step 5: 验证特征质量

**查看 Cell 7 数据质量报告**：
```
📊 Missing values:
   ✅ No missing values!

📊 Label distribution:
   Positive (1): 62 (71.3%)
   Negative (0): 25 (28.7%)

📊 Feature statistics:
              flux_mean    flux_std  bls_power  ...
count            87.00       87.00      87.00
mean              1.00        0.02       0.15
std               0.00        0.01       0.08
min               0.99        0.00       0.05
max               1.01        0.05       0.42

🔍 Checking for infinities:
   ✅ No infinities or extreme values
```

**如果有问题**：
```python
# 检查异常特征
import pandas as pd
features_df = pd.read_parquet('checkpoints/features_checkpoint.parquet')

# 查看缺失值
print(features_df.isnull().sum())

# 查看极值
print(features_df.describe())

# 检查 BLS 失败样本
failed_bls = features_df[features_df['bls_power'] == 0.0]
print(f"BLS 失败样本: {len(failed_bls)}")
```

---

## ✅ 测试通过标准

**必须满足**：
- [x] 成功下载 >80 个样本（85%+ 成功率）
- [x] 文件大小正常（3-6 MB/文件）
- [x] 特征提取无错误
- [x] 无缺失值或无穷值
- [x] BLS 特征合理（power > 0）

**如果测试失败**：
1. 检查网络连接
2. 检查 MAST 服务状态: https://mast.stsci.edu/
3. 查看错误日志：`checkpoints/download_report.json`
4. 在 GitHub Issues 报告问题

---

## 🚀 测试通过后：全量下载

### Step 6: 配置全量下载

**修改 `02a_download_lightcurves.ipynb` Cell 4**：
```python
CONFIG = {
    'max_workers': 6,        # ⬆️ 提高到 6（本地网络稳定）
    'max_retries': 3,
    'timeout': 60,
    'batch_size': 100,
    'save_interval': 50,
    'test_mode': False,      # ⬇️ 改为 False（全量下载）
}
```

**清理测试数据（可选）**：
```bash
# 如果要重新开始（删除测试文件）
rm -rf data/lightcurves/*.pkl
rm -f checkpoints/download_progress.parquet
rm -f checkpoints/download_report.json
```

**或者保留测试数据**：
```python
# 02a 会自动跳过已下载的文件
# 测试的 100 个样本不会重复下载
```

---

### Step 7: 启动全量下载

**最佳时机**：
```
建议时间：晚上 10:00 PM 启动
完成时间：次日早上 6:00 AM
```

**执行**：
```python
# 在 Jupyter Notebook
1. 确认 Cell 4 配置正确
2. Cell → Run All
3. 检查进度条启动
4. 关闭笔记本屏幕（不要关机）
```

**预计统计**：
```
🚀 Starting download for 11,879 samples
   Workers: 6
   Estimated time: 5.5 hours
```

**监控脚本**（可选后台运行）：
```bash
# monitor_download.sh
#!/bin/bash
while true; do
    count=$(ls data/lightcurves/*.pkl 2>/dev/null | wc -l)
    echo "[$(date +%H:%M)] Downloaded: $count / 11979"
    sleep 300  # 每 5 分钟检查
done
```

---

### Step 8: 全量特征提取

**次日早上检查下载完成后**：

1. 打开 `02b_extract_features.ipynb`
2. 无需修改配置
3. Run All（预计 15-20 分钟）

**输出示例**：
```
✅ Feature extraction complete
   Total features: 10,234
   Feature columns: 14
   Success rate: 85.6%
```

---

### Step 9: 开始训练

**修改 `03_injection_train_PRODUCTION.ipynb`**：

找到 **Cell 6**，替换为：
```python
# 从文件加载特征（不再下载）
features_path = MODEL_DIR / 'features_20250104_080000.parquet'  # 使用实际文件名
features_df = pd.read_parquet(features_path)

print(f"✅ Loaded {len(features_df):,} features from disk")
print(f"   Positive: {features_df['label'].sum():,}")
print(f"   Negative: {(~features_df['label'].astype(bool)).sum():,}")

# 定义特征列
feature_cols = [col for col in features_df.columns
                if col not in ['sample_id', 'tic_id', 'label', 'n_sectors']]

print(f"   Features: {len(feature_cols)}")
```

**然后直接跳到 Cell 7 训练模型**

---

## 📊 时间线总结

| 阶段 | 时间 | 描述 |
|------|------|------|
| 测试下载 | 15-20 min | 100 样本验证流程 |
| 测试特征提取 | 2-3 min | 验证特征质量 |
| **测试总计** | **~20 min** | **确保流程可行** |
| 全量下载 | 5-7 hours | 11,979 样本（晚上运行） |
| 全量特征提取 | 15-20 min | 提取所有特征 |
| **全量总计** | **~6 hours** | **一次性投资** |

---

## 🆘 常见问题

### Q1: 下载速度慢怎么办？
```python
# 降低并发数（避免被限速）
CONFIG = {'max_workers': 2}  # 改为 2

# 或增加超时时间
CONFIG = {'timeout': 120}  # 改为 120 秒
```

### Q2: 某些样本找不到数据？
```
正常现象！并非所有 TOI 都有 SPOC 光曲线
预期成功率：85-90%
失败原因：no_data_found, download_failed
```

### Q3: 硬盘空间不够？
```bash
# 检查当前使用
du -sh data/lightcurves/

# 预计全量大小：
# 100 样本  ≈ 400 MB
# 11,979 样本 ≈ 48 GB

# 如果空间不足，删除测试文件
rm data/lightcurves/SAMPLE_0000*.pkl
```

### Q4: 中途停止了怎么办？
```python
# 重新运行 02a Cell 6
# 自动从 checkpoint 恢复
# 已下载的样本会跳过
```

---

## ✅ 检查清单

**测试阶段（今天）**：
- [ ] 修改 02a Cell 4 为 test_mode=True
- [ ] 成功下载 >80 个样本
- [ ] 验证文件完整性（Cell 7）
- [ ] 成功提取测试特征（02b）
- [ ] 检查特征质量无异常

**全量阶段（明天）**：
- [ ] 修改 02a Cell 4 为 test_mode=False, max_workers=6
- [ ] 晚上启动全量下载
- [ ] 次日早上检查完成（>10,000 文件）
- [ ] 运行 02b 提取全量特征
- [ ] 开始训练模型

---

## 📞 需要帮助？

- 查看日志：`checkpoints/download_report.json`
- 检查错误：`progress_df[progress_df['status']=='failed']`
- GitHub Issues: https://github.com/exoplanet-spaceapps/exoplanet-starter/issues

**祝测试顺利！🚀**
