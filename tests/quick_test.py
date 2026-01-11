"""
Quick test script to verify framework installation
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    try:
        from visionframework import Detector, Tracker, VisionPipeline, Visualizer, Config
        print("[OK] All imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False

def test_config():
    """Test configuration utilities"""
    print("\nTesting configuration...")
    try:
        from visionframework import Config
        
        detector_config = Config.get_default_detector_config()
        tracker_config = Config.get_default_tracker_config()
        pipeline_config = Config.get_default_pipeline_config()
        
        print("[OK] Configuration utilities working")
        print(f"  - Detector config keys: {list(detector_config.keys())}")
        print(f"  - Tracker config keys: {list(tracker_config.keys())}")
        return True
    except Exception as e:
        print(f"[FAIL] Configuration test failed: {e}")
        return False

def test_initialization():
    """Test module initialization"""
    print("\nTesting module initialization...")
    try:
        from visionframework import Detector, Tracker
        
        # Test detector initialization (without actually loading model)
        detector = Detector({"model_path": "yolov8n.pt"})
        print("[OK] Detector created")
        
        # Test tracker initialization
        tracker = Tracker()
        if tracker.initialize():
            print("[OK] Tracker initialized")
        else:
            print("[FAIL] Tracker initialization failed")
            return False
        
        return True
    except Exception as e:
        print(f"[FAIL] Initialization test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Vision Framework - Quick Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config,
        test_initialization
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    if all(results):
        print("[OK] All tests passed!")
        print("\nFramework is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Check examples/ directory for usage examples")
        print("3. Read README.md or docs/QUICKSTART.md for documentation")
    else:
        print("[FAIL] Some tests failed")
        print("Please check the error messages above")
    print("=" * 50)

if __name__ == "__main__":
    main()

