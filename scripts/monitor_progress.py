"""
即時監控 BLS 分析進度
"""
import pandas as pd
import os
import glob
from datetime import datetime
import time

data_dir = r"C:\Users\tingy\Desktop\dev\exoplanet-starter\data"
total_samples = 11979

def get_latest_checkpoint():
    """獲取最新的檢查點檔案"""
    checkpoint_files = glob.glob(os.path.join(data_dir, "bls_results_checkpoint_*.csv"))
    if checkpoint_files:
        latest = max(checkpoint_files, key=os.path.getctime)
        return latest
    return None

def check_results():
    """檢查結果檔案"""
    results_file = os.path.join(data_dir, "bls_results.csv")
    if os.path.exists(results_file):
        return results_file
    return None

def get_progress():
    """獲取當前進度"""
    # 優先檢查最終結果
    results_file = check_results()
    if results_file:
        df = pd.read_csv(results_file)
        return len(df), results_file, True

    # 檢查最新檢查點
    checkpoint = get_latest_checkpoint()
    if checkpoint:
        df = pd.read_csv(checkpoint)
        return len(df), checkpoint, False

    return 0, None, False

def display_progress():
    """顯示進度資訊"""
    print("="*80)
    print("BLS Analysis Progress Monitor")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    processed, file_path, is_complete = get_progress()

    if processed == 0:
        print("[WAITING] Processing not started or not reached first checkpoint (10 samples)")
        print("   Please wait...")
    else:
        percentage = (processed / total_samples) * 100
        print(f"[PROCESSED] {processed:,} / {total_samples:,} samples ({percentage:.2f}%)")
        print(f"[FILE] {os.path.basename(file_path)}")

        if is_complete:
            print("[COMPLETE] Processing finished!")
        else:
            remaining = total_samples - processed
            print(f"[REMAINING] {remaining:,} samples")

            # 估計完成時間（假設平均 8 秒/樣本）
            avg_time_per_sample = 8  # 秒
            estimated_seconds = remaining * avg_time_per_sample
            estimated_hours = estimated_seconds / 3600
            estimated_days = estimated_hours / 24

            if estimated_days > 1:
                print(f"[ESTIMATED] ~{estimated_days:.1f} days to complete")
            else:
                print(f"[ESTIMATED] ~{estimated_hours:.1f} hours to complete")

        # 顯示一些統計
        if file_path:
            df = pd.read_csv(file_path)
            if 'success' in df.columns:
                success_count = df['success'].sum()
                fail_count = len(df) - success_count
                success_rate = (success_count / len(df)) * 100
                print()
                print(f"[SUCCESS] {success_count:,} samples ({success_rate:.1f}%)")
                print(f"[FAILED] {fail_count:,} samples")

    print()
    print("="*80)
    print("Tips:")
    print("   - Auto-saves checkpoint every 10 samples")
    print("   - Can resume from checkpoint if interrupted")
    print("   - Full processing takes ~3-5 days (continuous)")
    print("="*80)

if __name__ == "__main__":
    # 持續監控模式
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        print("[WATCH MODE] Continuous monitoring (Press Ctrl+C to stop)\n")
        try:
            while True:
                display_progress()
                print("\n[WAITING] Updating in 30 seconds...\n")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n\n[STOPPED] Monitor stopped")
    else:
        display_progress()
        print("\nTip: Use 'python monitor_progress.py --watch' for continuous monitoring")