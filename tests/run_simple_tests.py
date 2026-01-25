#!/usr/bin/env python3
"""
Run simple tests that don't depend on torch or ultralytics.

This script runs only the tests that don't require the problematic torch/ultralytics imports.
"""

import pytest
import sys
import os


def main():
    """
    Run simple tests that don't depend on torch or ultralytics.
    
    Returns:
        int: Exit code (0 if all tests passed, 1 otherwise)
    """
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of specific test cases that don't depend on torch/ultralytics
    simple_tests = [
        # TestConfigUtils tests
        "test_utils.py::TestConfigUtils::test_load_save_config",
        "test_utils.py::TestConfigUtils::test_load_nonexistent_config",
        # TestImageUtils tests
        "test_utils.py::TestImageUtils::test_resize_image",
        "test_utils.py::TestImageUtils::test_load_save_image",
        # TestPerformanceUtils tests
        "test_utils.py::TestPerformanceUtils::test_performance_monitor",
        "test_utils.py::TestPerformanceUtils::test_performance_monitor_basic",
        # TestLoggingExceptions tests from test_misc.py
        "test_misc.py::TestLoggingExceptions::test_get_logger",
        "test_misc.py::TestLoggingExceptions::test_logger_levels",
        # TestExceptions tests from test_misc.py
        "test_misc.py::TestExceptions::test_vision_framework_error",
        "test_misc.py::TestExceptions::test_configuration_error",
        "test_misc.py::TestExceptions::test_device_error",
        "test_misc.py::TestExceptions::test_exception_hierarchy",
        # TestCodeQuality tests from test_misc.py (simple ones that don't import torch)
        "test_misc.py::TestCodeQuality::test_module_structure"
    ]
    
    # Run each test individually
    all_passed = True
    for test_case in simple_tests:
        print(f"\n=== Running test: {test_case} ===")
        try:
            result = pytest.main([os.path.join(test_dir, test_case), "-v"])
            if result != 0:
                all_passed = False
        except Exception as e:
            print(f"  ERROR running {test_case}: {e}")
            all_passed = False
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
