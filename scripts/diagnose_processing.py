"""
診斷處理狀態 - 檢查是否有樣本成功處理
"""
import os
import glob

data_dir = r"C:\Users\tingy\Desktop\dev\exoplanet-starter\data"

print("="*80)
print("Processing Diagnostics")
print("="*80)
print()

# 檢查所有 BLS 相關文件
print("[1] Checking for BLS result files...")
all_files = os.listdir(data_dir)
bls_files = [f for f in all_files if 'bls' in f.lower() or 'checkpoint' in f.lower()]

if bls_files:
    print(f"    Found {len(bls_files)} BLS-related files:")
    for f in bls_files:
        filepath = os.path.join(data_dir, f)
        size = os.path.getsize(filepath)
        print(f"    - {f} ({size:,} bytes)")
else:
    print("    [WARNING] No BLS result files found yet")
    print("    This is normal if:")
    print("      - Processing just started")
    print("      - Less than 10 samples processed successfully")
    print("      - Most samples failed to download")

print()
print("[2] Checking working directory...")
cwd = os.getcwd()
print(f"    Current directory: {cwd}")

# 檢查當前目錄的data子目錄
if os.path.exists("data"):
    print("    ./data exists")
    local_bls = glob.glob("data/*bls*.csv") + glob.glob("data/*checkpoint*.csv")
    if local_bls:
        print(f"    Found {len(local_bls)} files in ./data:")
        for f in local_bls:
            size = os.path.getsize(f)
            print(f"    - {f} ({size:,} bytes)")
else:
    print("    [WARNING] ./data does not exist in current directory")

print()
print("[3] Expected behavior:")
print("    - Checkpoint saved every 10 SUCCESSFUL samples")
print("    - Failed samples are skipped (common for TESS data)")
print("    - If all samples fail to download, no checkpoint is created")
print()
print("="*80)