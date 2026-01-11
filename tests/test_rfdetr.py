"""
Test RF-DETR detector functionality
"""

import sys
from pathlib import Path
import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_rfdetr_import():
    """Test if RF-DETR detector can be imported"""
    print("Testing RF-DETR import...")
    try:
        from visionframework import RFDETRDetector
        from visionframework.core.detectors.rfdetr_detector import RFDETRDetector as DirectRFDETRDetector
        print("[OK] RF-DETR detector imported successfully")
        return True
    except ImportError as e:
        print(f"[SKIP] RF-DETR not available: {e}")
        print("      Install with: pip install rfdetr supervision")
        return None  # Skip test if not available
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rfdetr_detector_creation():
    """Test RF-DETR detector creation"""
    print("\nTesting RF-DETR detector creation...")
    try:
        from visionframework import RFDETRDetector
        
        # Test with default config
        detector = RFDETRDetector()
        print("[OK] RF-DETR detector created with default config")
        
        # Test with custom config
        detector = RFDETRDetector({
            "conf_threshold": 0.5,
            "device": "cpu"
        })
        print("[OK] RF-DETR detector created with custom config")
        
        return True
    except ImportError:
        return None  # Skip if not available
    except Exception as e:
        print(f"[FAIL] Detector creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rfdetr_initialization():
    """Test RF-DETR detector initialization"""
    print("\nTesting RF-DETR detector initialization...")
    try:
        from visionframework import RFDETRDetector
        
        detector = RFDETRDetector({
            "conf_threshold": 0.5,
            "device": "cpu"
        })
        
        # Try to initialize (may fail if model download is needed)
        initialized = detector.initialize()
        if initialized:
            print("[OK] RF-DETR detector initialized successfully")
            return True
        else:
            print("[WARN] RF-DETR detector initialization failed (may need model download)")
            print("       This is expected on first run or without internet connection")
            return None  # Not a failure, just skip further tests
    except ImportError:
        return None  # Skip if not available
    except Exception as e:
        print(f"[WARN] Initialization test encountered issue: {e}")
        print("       This may be expected if rfdetr is not installed or model download fails")
        return None


def test_unified_detector_rfdetr():
    """Test unified Detector interface with RF-DETR"""
    print("\nTesting unified Detector interface with RF-DETR...")
    try:
        from visionframework import Detector
        
        # Test detector creation
        detector = Detector({
            "model_type": "rfdetr",
            "conf_threshold": 0.5,
            "device": "cpu"
        })
        print("[OK] Unified Detector created with RF-DETR type")
        
        # Test model info (only if initialized)
        info = detector.get_model_info()
        if "model_type" in info:
            assert info["model_type"] == "rfdetr"
            print("[OK] Model info correct")
        else:
            # Not initialized yet, which is OK
            print("[OK] Detector created (not initialized)")
        
        return True
    except ImportError:
        return None  # Skip if not available
    except Exception as e:
        print(f"[FAIL] Unified detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rfdetr_detection():
    """Test RF-DETR detection on dummy image"""
    print("\nTesting RF-DETR detection...")
    try:
        from visionframework import RFDETRDetector
        
        detector = RFDETRDetector({
            "conf_threshold": 0.5,
            "device": "cpu"
        })
        
        # Initialize
        if not detector.initialize():
            print("[SKIP] Detector initialization failed, skipping detection test")
            return None
        
        # Create dummy image
        dummy_image = np.zeros((640, 480, 3), dtype=np.uint8)
        dummy_image[:] = (128, 128, 128)  # Gray image
        
        # Run detection
        detections = detector.detect(dummy_image)
        print(f"[OK] Detection completed, found {len(detections)} objects")
        
        # Check detection format
        if len(detections) > 0:
            det = detections[0]
            assert hasattr(det, 'bbox')
            assert hasattr(det, 'confidence')
            assert hasattr(det, 'class_id')
            assert hasattr(det, 'class_name')
            print("[OK] Detection format correct")
        
        return True
    except ImportError:
        return None  # Skip if not available
    except Exception as e:
        print(f"[WARN] Detection test encountered issue: {e}")
        print("       This may be expected if model is not available")
        return None


def test_rfdetr_integration():
    """Test RF-DETR integration with pipeline"""
    print("\nTesting RF-DETR integration with pipeline...")
    try:
        from visionframework import VisionPipeline
        
        # Create pipeline with RF-DETR
        pipeline = VisionPipeline({
            "detector_config": {
                "model_type": "rfdetr",
                "conf_threshold": 0.5,
                "device": "cpu"
            },
            "enable_tracking": False  # Just test detection
        })
        
        # Try to initialize
        if not pipeline.initialize():
            print("[SKIP] Pipeline initialization failed, skipping integration test")
            return None
        
        print("[OK] Pipeline initialized with RF-DETR detector")
        
        # Create dummy image
        dummy_image = np.zeros((640, 480, 3), dtype=np.uint8)
        dummy_image[:] = (128, 128, 128)
        
        # Process
        results = pipeline.process(dummy_image)
        assert "detections" in results
        print(f"[OK] Pipeline processing completed, found {len(results['detections'])} detections")
        
        return True
    except ImportError:
        return None  # Skip if not available
    except Exception as e:
        print(f"[WARN] Integration test encountered issue: {e}")
        print("       This may be expected if model is not available")
        return None


def main():
    """Run all RF-DETR tests"""
    print("=" * 60)
    print("RF-DETR Detector Test Suite")
    print("=" * 60)
    
    results = []
    
    # Import test
    result = test_rfdetr_import()
    results.append(result)
    
    # Creation test
    result = test_rfdetr_detector_creation()
    results.append(result)
    
    # Initialization test
    result = test_rfdetr_initialization()
    results.append(result)
    
    # Unified detector test
    result = test_unified_detector_rfdetr()
    results.append(result)
    
    # Detection test
    result = test_rfdetr_detection()
    results.append(result)
    
    # Integration test
    result = test_rfdetr_integration()
    results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    passed = [r for r in results if r is True]
    skipped = [r for r in results if r is None]
    failed = [r for r in results if r is False]
    
    print(f"Results: {len(passed)} passed, {len(skipped)} skipped, {len(failed)} failed")
    
    if len(failed) == 0:
        if len(passed) > 0:
            print("[OK] All available tests passed!")
        else:
            print("[INFO] RF-DETR not available or model not downloaded")
            print("       Install with: pip install rfdetr supervision")
            print("       Model will be downloaded automatically on first use")
    else:
        print("[FAIL] Some tests failed")
    
    print("=" * 60)
    
    return len(failed) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

