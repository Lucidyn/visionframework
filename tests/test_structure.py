"""
Test framework structure without importing heavy dependencies
"""

import sys
from pathlib import Path

# Get project root directory (parent of tests/)
PROJECT_ROOT = Path(__file__).parent.parent

def test_file_structure():
    """Test if all required files exist"""
    print("Testing file structure...")
    
    required_files = [
        # Core package files
        "visionframework/__init__.py",
        "visionframework/core/__init__.py",
        "visionframework/core/base.py",
        "visionframework/core/detector.py",
        "visionframework/core/tracker.py",
        "visionframework/core/pipeline.py",
        "visionframework/core/roi_detector.py",
        "visionframework/core/counter.py",
        # Data structures
        "visionframework/data/__init__.py",
        "visionframework/data/detection.py",
        "visionframework/data/track.py",
        "visionframework/data/pose.py",
        "visionframework/data/roi.py",
        # Detector implementations
        "visionframework/core/detectors/__init__.py",
        "visionframework/core/detectors/base_detector.py",
        "visionframework/core/detectors/yolo_detector.py",
        "visionframework/core/detectors/detr_detector.py",
        "visionframework/core/detectors/rfdetr_detector.py",
        # Tracker implementations
        "visionframework/core/trackers/__init__.py",
        "visionframework/core/trackers/base_tracker.py",
        "visionframework/core/trackers/iou_tracker.py",
        "visionframework/core/trackers/byte_tracker.py",
        # Pose estimator
        "visionframework/core/pose_estimator.py",
        # Utils files
        "visionframework/utils/__init__.py",
        "visionframework/utils/config.py",
        "visionframework/utils/image_utils.py",
        "visionframework/utils/export.py",
        "visionframework/utils/performance.py",
        "visionframework/utils/video_utils.py",
        "visionframework/utils/logger.py",
        "visionframework/utils/trajectory_analyzer.py",
        # Visualization submodule
        "visionframework/utils/visualization/__init__.py",
        "visionframework/utils/visualization/unified_visualizer.py",
        # Evaluation submodule
        "visionframework/utils/evaluation/__init__.py",
        "visionframework/utils/evaluation/detection_evaluator.py",
        "visionframework/utils/evaluation/tracking_evaluator.py",
        # Project files
        "requirements.txt",
        "setup.py",
        "README.md",
        # Documentation
        "docs/QUICKSTART.md",
        "docs/FEATURES.md",
        "docs/CHANGELOG.md",
        # Examples
        "examples/basic_usage.py",
        "examples/video_tracking.py",
        "examples/advanced_features.py",
        "examples/config_example.py",
        "examples/rfdetr_example.py",
        "examples/rfdetr_tracking.py",
        "examples/yolo_pose_example.py",
        "examples/batch_processing.py",
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [FAIL] {file_path} not found")
            all_exist = False
    
    return all_exist

def test_module_structure():
    """Test if modules can be parsed (syntax check)"""
    print("\nTesting module structure...")
    
    try:
        import ast
        
        # Test core modules (no heavy dependencies)
        core_modules = [
            "visionframework/core/base.py",
            "visionframework/core/roi_detector.py",
            "visionframework/core/counter.py",
        ]
        
        for module_path in core_modules:
            full_path = PROJECT_ROOT / module_path
            with open(full_path, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            print(f"  [OK] {module_path} syntax valid")
        
        # Test utils modules (no heavy dependencies)
        utils_modules = [
            "visionframework/utils/config.py",
            "visionframework/utils/export.py",
            "visionframework/utils/performance.py",
        ]
        
        for module_path in utils_modules:
            full_path = PROJECT_ROOT / module_path
            with open(full_path, 'r', encoding='utf-8') as f:
                ast.parse(f.read())
            print(f"  [OK] {module_path} syntax valid")
        
        return True
    except SyntaxError as e:
        print(f"  [FAIL] Syntax error: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False

def main():
    """Run structure tests"""
    print("=" * 50)
    print("Vision Framework - Structure Test")
    print("=" * 50)
    
    results = [
        test_file_structure(),
        test_module_structure()
    ]
    
    print("\n" + "=" * 50)
    if all(results):
        print("[OK] Framework structure is correct!")
        print("\nFramework files are ready.")
        print("\nNote: To use the framework, you need to:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Make sure numpy < 2.0.0 for compatibility")
        print("3. Check examples/ directory for usage")
        print("4. Read docs/QUICKSTART.md for quick start guide")
    else:
        print("[FAIL] Some structure tests failed")
    print("=" * 50)

if __name__ == "__main__":
    main()

