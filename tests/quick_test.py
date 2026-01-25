#!/usr/bin/env python3
"""
Quick test script to verify the Vision Framework is working correctly.

This script runs basic functionality tests to ensure the framework is
properly installed and functioning.
"""

import os
import sys

# Set environment variable to avoid OpenMP runtime issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import cv2

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionframework.core.pipeline import VisionPipeline
from visionframework.utils.config import DeviceManager


def test_device_selection():
    """Test device selection functionality"""
    print("Testing device selection...")
    best_device = DeviceManager.auto_select_device()
    print(f"  Best device: {best_device}")
    return True


def test_basic_detection():
    """Test basic detection functionality"""
    print("Testing basic detection...")
    
    # Create a simple test image with a white rectangle
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (200, 200), (400, 400), (255, 255, 255), -1)
    
    try:
        # Initialize pipeline with a small model for quick testing
        pipeline = VisionPipeline({
            "detector_config": {
                "model_path": "yolov8n.pt",
                "conf_threshold": 0.25
            }
        })
        
        # Process the image
        results = pipeline.process(image)
        
        print(f"  Detection results: {len(results['detections'])} objects detected")
        for det in results['detections']:
            print(f"    - {det['class_name']}: {det['confidence']:.2f}")
        
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_simplified_api():
    """Test simplified API functionality"""
    print("Testing simplified API...")
    
    # Create a simple test image
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (200, 200), (400, 400), (255, 255, 255), -1)
    
    try:
        # Use the static method for quick processing
        results = VisionPipeline.process_image(
            image=image, 
            model_path="yolov8n.pt",
            enable_tracking=False,
            conf_threshold=0.25
        )
        
        print(f"  Simplified API results: {len(results['detections'])} objects detected")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def test_tracking():
    """Test basic tracking functionality"""
    print("Testing tracking...")
    
    # Create a simple test image
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (200, 200), (400, 400), (255, 255, 255), -1)
    
    try:
        # Initialize pipeline with tracking
        pipeline = VisionPipeline({
            "enable_tracking": True,
            "detector_config": {
                "model_path": "yolov8n.pt",
                "conf_threshold": 0.25
            },
            "tracker_config": {
                "tracker_type": "iou"
            }
        })
        
        # Process the image
        results = pipeline.process(image)
        
        # Check that results contains tracks key (even if empty)
        assert "tracks" in results
        print(f"  Tracking results: Pipeline processed image successfully with {len(results['tracks'])} tracks")
        
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    """Run all quick tests"""
    print("=" * 50)
    print("VISION FRAMEWORK QUICK TEST")
    print("=" * 50)
    
    tests = [
        ("Device Selection", test_device_selection),
        ("Basic Detection", test_basic_detection),
        ("Simplified API", test_simplified_api),
        ("Tracking", test_tracking)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        if test_func():
            passed += 1
            print("✓ PASSED")
        else:
            print("✗ FAILED")
    
    print("\n" + "=" * 50)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed! The Vision Framework is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
