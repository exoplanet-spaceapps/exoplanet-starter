#!/usr/bin/env python3
"""
Test Runner Script for Exoplanet Starter Project

Quick way to run all tests with coverage reporting.

Usage:
    python tests/run_tests.py              # Run all tests
    python tests/run_tests.py --fast       # Run fast tests only
    python tests/run_tests.py --verbose    # Verbose output
    python tests/run_tests.py --coverage   # With coverage report
"""

import sys
import subprocess
from pathlib import Path


def run_tests(args=None):
    """Run pytest with specified arguments."""
    if args is None:
        args = []

    # Base pytest command
    cmd = ['pytest', 'tests/test_notebook_02.py']

    # Parse command line arguments
    if '--fast' in sys.argv:
        cmd.extend(['-m', 'not slow'])
        print("🏃 Running fast tests only...")
    else:
        print("🧪 Running all tests...")

    if '--verbose' in sys.argv or '-v' in sys.argv:
        cmd.append('-v')
    else:
        cmd.append('-v')  # Always verbose by default

    if '--coverage' in sys.argv or '--cov' in sys.argv:
        cmd.extend(['--cov=.', '--cov-report=html', '--cov-report=term-missing'])
        print("📊 Coverage reporting enabled...")

    # Always show short traceback
    cmd.append('--tb=short')

    # Add timeout
    cmd.extend(['--timeout=60'])

    print(f"\n📋 Command: {' '.join(cmd)}\n")
    print("=" * 60)

    # Run tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except FileNotFoundError:
        print("\n❌ ERROR: pytest not found!")
        print("Install test dependencies with:")
        print("  pip install -r tests/requirements-test.txt")
        return 1
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        return 130


def main():
    """Main entry point."""
    print("=" * 60)
    print("🚀 EXOPLANET STARTER - TEST SUITE RUNNER")
    print("=" * 60)
    print()

    # Check if test file exists
    test_file = Path('tests/test_notebook_02.py')
    if not test_file.exists():
        print(f"❌ ERROR: Test file not found: {test_file}")
        return 1

    # Run tests
    exit_code = run_tests()

    # Print summary
    print()
    print("=" * 60)
    if exit_code == 0:
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print()
        print("📊 Next steps:")
        print("  1. Review coverage report: htmlcov/index.html")
        print("  2. Add test cells to notebook from: docs/notebook_02_test_cells.md")
        print("  3. Run notebook tests in Google Colab")
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        print()
        print("🔍 Troubleshooting:")
        print("  1. Check test output above for details")
        print("  2. Review tests/README_TESTS.md for common issues")
        print("  3. Ensure all dependencies are installed")

    return exit_code


if __name__ == '__main__':
    sys.exit(main())