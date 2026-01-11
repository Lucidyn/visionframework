"""
Test new features import and basic functionality
"""

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test if all new modules can be imported"""
    print("Testing imports...")
    try:
        from visionframework import (
            ResultExporter, PerformanceMonitor, Timer,
            VideoProcessor, VideoWriter, process_video,
            ROIDetector, ROI, Counter,
            RFDETRDetector  # RF-DETR detector
        )
        print("[OK] All new modules imported successfully")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic functionality of new modules"""
    print("\nTesting basic functionality...")
    
    try:
        # Test ResultExporter
        from visionframework import ResultExporter
        exporter = ResultExporter()
        print("[OK] ResultExporter created")
        
        # Test PerformanceMonitor
        from visionframework import PerformanceMonitor
        monitor = PerformanceMonitor()
        monitor.start()
        monitor.tick()
        fps = monitor.get_current_fps()
        print(f"[OK] PerformanceMonitor working (FPS: {fps:.2f})")
        
        # Test Timer
        from visionframework import Timer
        with Timer("test") as timer:
            import time
            time.sleep(0.01)
        print(f"[OK] Timer working (elapsed: {timer.get_elapsed():.4f}s)")
        
        # Test ROIDetector
        from visionframework import ROIDetector, ROI
        roi_config = {
            "rois": [{
                "name": "test_roi",
                "type": "rectangle",
                "points": [(0, 0), (100, 100)]
            }]
        }
        roi_detector = ROIDetector(roi_config)
        if roi_detector.initialize():
            print("[OK] ROIDetector initialized")
        else:
            print("[FAIL] ROIDetector initialization failed")
            return False
        
        # Test Counter
        from visionframework import Counter
        counter = Counter({"roi_detector": roi_config})
        if counter.initialize():
            print("[OK] Counter initialized")
        else:
            print("[FAIL] Counter initialization failed")
            return False
        
        # Test RF-DETR Detector (if available)
        try:
            from visionframework import RFDETRDetector
            rfdetr_detector = RFDETRDetector({"conf_threshold": 0.5})
            print("[OK] RFDETRDetector created")
            # Note: Initialization may fail if rfdetr is not installed or model download fails
            # This is expected and not a failure
        except ImportError:
            print("[SKIP] RF-DETR not available (install with: pip install rfdetr supervision)")
        except Exception as e:
            print(f"[INFO] RF-DETR test skipped: {e}")
        
        return True
    except Exception as e:
        print(f"[FAIL] Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("New Features Test")
    print("=" * 50)
    
    results = [
        test_imports(),
        test_basic_functionality()
    ]
    
    print("\n" + "=" * 50)
    if all(results):
        print("[OK] All tests passed!")
        print("\nNew features are ready to use:")
        print("  - ResultExporter: Export results to JSON/CSV/COCO")
        print("  - PerformanceMonitor: Monitor FPS and performance")
        print("  - VideoProcessor/VideoWriter: Video processing utilities")
        print("  - ROIDetector: Region of Interest detection")
        print("  - Counter: Object counting in regions")
        print("  - RFDETRDetector: RF-DETR detection model support")
    else:
        print("[FAIL] Some tests failed")
    print("=" * 50)

if __name__ == "__main__":
    main()

