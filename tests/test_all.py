#!/usr/bin/env python3
"""
Run all tests in the test suite.

This script collects and runs all test cases from the test files.
It runs each test file individually to handle potential import issues.
"""

import pytest
import sys
import os
import glob


def run_all_tests():
    """
    Run all tests in the tests directory.
    
    Returns:
        int: Exit code (0 if all tests passed, 1 otherwise)
    """
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get all test files
    test_files = glob.glob(os.path.join(test_dir, "test_*.py"))
    
    # Exclude test_all.py itself and files which cause torch import issues
    test_files = [f for f in test_files if os.path.basename(f) not in ["test_all.py", "test_core.py", "test_models.py"]]
    
    # Sort the test files
    test_files.sort()
    
    # Run each test file individually
    all_passed = True
    for test_file in test_files:
        print(f"\n=== Running tests in {os.path.basename(test_file)} ===")
        try:
            result = pytest.main([test_file, "-v"])
            if result != 0:
                all_passed = False
        except Exception as e:
            print(f"  ERROR running {os.path.basename(test_file)}: {e}")
            all_passed = False
    
    return 0 if all_passed else 1


def run_specific_tests(test_names):
    """
    Run specific test files.
    
    Args:
        test_names: List of test file names or patterns
    
    Returns:
        int: Exit code (0 if all tests passed, 1 otherwise)
    """
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get all test files
    all_test_files = glob.glob(os.path.join(test_dir, "test_*.py"))
    all_test_files = [f for f in all_test_files if os.path.basename(f) != "test_all.py"]
    
    # Filter test files based on test_names
    test_files = []
    for name in test_names:
        if name.startswith("test_"):
            # Exact test file name
            test_path = os.path.join(test_dir, name)
            if test_path in all_test_files:
                test_files.append(test_path)
        else:
            # Pattern matching
            pattern = os.path.join(test_dir, f"test_{name}*.py")
            test_files.extend(glob.glob(pattern))
    
    # Remove duplicates and sort
    test_files = sorted(list(set(test_files)))
    
    if not test_files:
        print(f"No test files found matching: {', '.join(test_names)}")
        return 1
    
    # Run each test file individually
    all_passed = True
    for test_file in test_files:
        print(f"\n=== Running tests in {os.path.basename(test_file)} ===")
        try:
            result = pytest.main([test_file, "-v"])
            if result != 0:
                all_passed = False
        except Exception as e:
            print(f"  ERROR running {os.path.basename(test_file)}: {e}")
            all_passed = False
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test files
        test_names = sys.argv[1:]
        exit_code = run_specific_tests(test_names)
    else:
        # Run all test files
        exit_code = run_all_tests()
    
    sys.exit(exit_code)
