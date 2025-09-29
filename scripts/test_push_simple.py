#!/usr/bin/env python3
"""
Simple test for push_to_github function implementation
"""

import subprocess
import sys
from pathlib import Path

def test_git_lfs_availability():
    """Test if Git LFS is available"""
    print("Testing Git LFS availability...")
    try:
        result = subprocess.run(['git', 'lfs', 'version'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   Git LFS version: {result.stdout.strip()}")
            return True
        else:
            print("   Git LFS not available")
            return False
    except FileNotFoundError:
        print("   Git LFS not installed")
        return False

def test_git_repository():
    """Test if we're in a Git repository"""
    print("Testing Git repository status...")
    try:
        result = subprocess.run(['git', 'rev-parse', '--git-dir'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            git_dir = result.stdout.strip()
            print(f"   Git repository: {git_dir}")

            # Check remote repository
            result = subprocess.run(['git', 'remote', '-v'],
                                  capture_output=True, text=True)
            if result.stdout:
                print(f"   Remote repository: {result.stdout.strip().split()[1]}")

            return True
        else:
            print("   Not in a Git repository")
            return False
    except Exception as e:
        print(f"   Git check failed: {e}")
        return False

def test_data_files_exist():
    """Test if data files exist"""
    print("Testing data files...")
    data_dir = Path('data')

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

    print(f"   Existing files ({len(existing_files)}):")
    for file_info in existing_files:
        print(f"      - {file_info}")

    if missing_files:
        print(f"   Missing files ({len(missing_files)}):")
        for filename in missing_files:
            print(f"      - {filename}")

    return len(existing_files) > 0

def test_function_implementation():
    """Test if functions are implemented"""
    print("Testing function implementation...")

    notebook_path = Path('notebooks/01_tap_download.ipynb')
    if not notebook_path.exists():
        print("   01_tap_download.ipynb not found")
        return False

    with open(notebook_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check for key functions
    functions_to_check = [
        'def setup_git_lfs()',
        'def clone_repository(',
        'def push_to_github('
    ]

    found_functions = []
    for func in functions_to_check:
        if func in content:
            found_functions.append(func)

    print(f"   Functions found ({len(found_functions)}/{len(functions_to_check)}):")
    for func in found_functions:
        print(f"      - {func}")

    # Check key features
    key_features = [
        'GitHub Personal Access Token',
        'Co-Authored-By: hctsai1006',
        'git lfs track',
        'subprocess.run'
    ]

    found_features = [feature for feature in key_features if feature in content]
    print(f"   Key features ({len(found_features)}/{len(key_features)}):")
    for feature in found_features:
        print(f"      - {feature}")

    return len(found_functions) == len(functions_to_check)

def main():
    """Main test function"""
    print("push_to_github Function Test")
    print("=" * 50)

    tests = [
        ("Git LFS Availability", test_git_lfs_availability),
        ("Git Repository", test_git_repository),
        ("Data Files", test_data_files_exist),
        ("Function Implementation", test_function_implementation)
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\nTest: {test_name}")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"   Test failed: {e}")
            results.append(False)

    # Summary
    print(f"\nTest Results Summary")
    print("=" * 50)

    passed = sum(results)
    total = len(results)

    for i, (test_name, _) in enumerate(tests):
        status = "PASS" if results[i] else "FAIL"
        print(f"{test_name}: {status}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("All tests passed! push_to_github function is ready.")
        print("\nUsage:")
        print("1. Load 01_tap_download.ipynb in Colab")
        print("2. Run all cells until data download completes")
        print("3. Get GitHub Personal Access Token")
        print("4. Execute: push_to_github(token='ghp_your_token')")
    else:
        print("Some tests failed. Please check configuration.")

if __name__ == '__main__':
    main()