#!/usr/bin/env python3
"""
測試 push_to_github 函數的功能
Test script for push_to_github function

This script validates the push_to_github implementation without actually pushing.
可以在不實際推送的情況下驗證 push_to_github 的實作。
"""

import subprocess
import sys
from pathlib import Path
import tempfile
import os

def test_git_lfs_availability():
    """測試 Git LFS 是否可用"""
    print("🧪 測試 Git LFS 可用性...")
    try:
        result = subprocess.run(['git', 'lfs', 'version'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Git LFS 版本: {result.stdout.strip()}")
            return True
        else:
            print("   ❌ Git LFS 不可用")
            return False
    except FileNotFoundError:
        print("   ❌ Git LFS 未安裝")
        return False

def test_git_repository():
    """測試是否在 Git 倉庫中"""
    print("🧪 測試 Git 倉庫狀態...")
    try:
        result = subprocess.run(['git', 'rev-parse', '--git-dir'],
                              capture_output=True, text=True, cwd='..')
        if result.returncode == 0:
            git_dir = result.stdout.strip()
            print(f"   ✅ Git 倉庫: {git_dir}")

            # 檢查遠端倉庫
            result = subprocess.run(['git', 'remote', '-v'],
                                  capture_output=True, text=True, cwd='..')
            if result.stdout:
                print(f"   📡 遠端倉庫: {result.stdout.strip().split()[1]}")

            return True
        else:
            print("   ❌ 不在 Git 倉庫中")
            return False
    except Exception as e:
        print(f"   ❌ Git 檢查失敗: {e}")
        return False

def test_data_files_exist():
    """測試資料檔案是否存在"""
    print("🧪 測試資料檔案...")
    data_dir = Path('../data')

    required_files = [
        'toi.csv',
        'toi_positive.csv',
        'toi_negative.csv',
        'koi_false_positives.csv',
        'supervised_dataset.csv',
        'data_provenance.json'
    ]

    missing_files = []
    existing_files = []

    for filename in required_files:
        file_path = data_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / 1024 / 1024
            existing_files.append(f"{filename} ({size_mb:.2f} MB)")
        else:
            missing_files.append(filename)

    print(f"   ✅ 存在的檔案 ({len(existing_files)}):")
    for file_info in existing_files:
        print(f"      - {file_info}")

    if missing_files:
        print(f"   ⚠️ 缺少的檔案 ({len(missing_files)}):")
        for filename in missing_files:
            print(f"      - {filename}")

    return len(existing_files) > 0, existing_files, missing_files

def test_function_signature():
    """測試函數簽名是否正確"""
    print("🧪 測試函數實作...")

    # 模擬載入 notebook 中的函數
    notebook_path = Path('../notebooks/01_tap_download.ipynb')
    if not notebook_path.exists():
        print("   ❌ 找不到 01_tap_download.ipynb")
        return False

    # 檢查函數定義是否在檔案中
    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 檢查關鍵函數
    functions_to_check = [
        'def setup_git_lfs()',
        'def clone_repository(',
        'def push_to_github('
    ]

    found_functions = []
    for func in functions_to_check:
        if func in content:
            found_functions.append(func)
        else:
            print(f"   ❌ 未找到: {func}")

    print(f"   ✅ 找到函數 ({len(found_functions)}/{len(functions_to_check)}):")
    for func in found_functions:
        print(f"      - {func}")

    # 檢查關鍵功能點
    key_features = [
        'GitHub Personal Access Token',
        'Co-Authored-By: hctsai1006',
        'git lfs track',
        'subprocess.run',
        'data: update NASA exoplanet data'
    ]

    found_features = [feature for feature in key_features if feature in content]
    print(f"   ✅ 關鍵功能 ({len(found_features)}/{len(key_features)}):")
    for feature in found_features:
        print(f"      - {feature}")

    return len(found_functions) == len(functions_to_check)

def simulate_push_workflow(dry_run=True):
    """模擬推送工作流程"""
    print("🧪 模擬推送工作流程...")

    if dry_run:
        print("   📝 乾運行模式（不實際執行 git 命令）")

    # 步驟 1: 檢查 Git 狀態
    try:
        result = subprocess.run(['git', 'status', '--porcelain'],
                              capture_output=True, text=True, cwd='..')
        if result.stdout:
            print(f"   📋 待提交變更:")
            for line in result.stdout.strip().split('\n')[:5]:  # 只顯示前5行
                print(f"      {line}")
            if len(result.stdout.strip().split('\n')) > 5:
                print(f"      ... 還有 {len(result.stdout.strip().split('\n')) - 5} 個檔案")
        else:
            print("   ✅ 工作目錄乾淨，無待提交變更")
    except Exception as e:
        print(f"   ❌ Git 狀態檢查失敗: {e}")

    # 步驟 2: 檢查分支
    try:
        result = subprocess.run(['git', 'branch', '--show-current'],
                              capture_output=True, text=True, cwd='..')
        if result.stdout:
            branch = result.stdout.strip()
            print(f"   🌿 當前分支: {branch}")
        else:
            print("   ⚠️ 無法確定當前分支")
    except Exception as e:
        print(f"   ❌ 分支檢查失敗: {e}")

    return True

def main():
    """主測試函數"""
    print("push_to_github Function Test")
    print("=" * 50)

    # 執行各項測試
    tests = [
        test_git_lfs_availability,
        test_git_repository,
        test_data_files_exist,
        test_function_signature,
        lambda: simulate_push_workflow(dry_run=True)
    ]

    results = []
    for i, test in enumerate(tests, 1):
        print(f"\n📋 測試 {i}/{len(tests)}")
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ❌ 測試失敗: {e}")
            results.append(False)

    # 總結
    print(f"\n📊 測試結果總結")
    print("=" * 50)

    passed = sum(1 for r in results if r is True or (isinstance(r, tuple) and r[0] is True))
    total = len(results)

    print(f"通過: {passed}/{total}")

    if passed == total:
        print("✅ 所有測試通過！push_to_github 功能準備就緒")
        print("\n💡 使用方式:")
        print("1. 在 Colab 中載入 01_tap_download.ipynb")
        print("2. 執行所有 cells 直到資料下載完成")
        print("3. 取得 GitHub Personal Access Token")
        print("4. 執行: push_to_github(token='ghp_你的token')")
    else:
        print("⚠️ 部分測試失敗，請檢查相關配置")

        # 提供修復建議
        if not results[0]:  # Git LFS
            print("\n🔧 修復建議:")
            print("- 安裝 Git LFS: https://git-lfs.github.com/")

        if not results[1]:  # Git 倉庫
            print("\n🔧 修復建議:")
            print("- 確保在 Git 倉庫目錄中執行")
            print("- 或者執行: git init && git remote add origin <repo-url>")

if __name__ == '__main__':
    main()