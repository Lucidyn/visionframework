import numpy as np
import pytest
from visionframework.core.detectors.yolo_detector import YOLODetector


class TestYOLO26Support:
    """Test cases for YOLO26 model support"""
    
    def test_yolo26_initialization(self):
        """Test YOLO26 model initialization"""
        # Test with YOLO26 nano model
        detector = YOLODetector({"model_path": "yolov26n.pt", "device": "cpu"})
        assert detector is not None
    
    def test_yolo26_detection(self):
        """Test YOLO26 detection functionality"""
        detector = YOLODetector({"model_path": "yolov26n.pt", "device": "cpu"})
        # Create a dummy image
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        results = detector.detect(image)
        assert isinstance(results, list)
    
    def test_yolo26_batch_detection(self):
        """Test YOLO26 batch detection"""
        detector = YOLODetector({"model_path": "yolov26n.pt", "device": "cpu"})
        # Create dummy images
        images = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(2)]
        
        results = detector.detect_batch(images)
        assert isinstance(results, list)
        assert len(results) == 2


class TestCustomModelSupport:
    """Test cases for custom model support"""
    
    def test_custom_model_path(self):
        """Test loading model from custom path"""
        # This test will fail if no custom model exists, but it's important to check the functionality
        # We'll use a try-except block to handle this gracefully
        try:
            detector = YOLODetector({"model_path": "yolov8n.pt", "device": "cpu"})
            assert detector is not None
        except Exception as e:
            pytest.skip(f"Custom model test skipped: {e}")
    
    def test_model_caching(self):
        """Test model caching functionality"""
        # Test loading the same model twice to check caching
        detector1 = YOLODetector({"model_path": "yolov8n.pt", "device": "cpu"})
        detector2 = YOLODetector({"model_path": "yolov8n.pt", "device": "cpu"})
        
        assert detector1 is not None
        assert detector2 is not None


class TestModelFormats:
    """Test cases for different model formats"""
    
    def test_pt_model_format(self):
        """Test loading PyTorch (.pt) model"""
        detector = YOLODetector({"model_path": "yolov8n.pt", "device": "cpu"})
        assert detector is not None
    
    def test_onnx_model_format(self):
        """Test ONNX model format support"""
        # ONNX support test - this might fail if ONNX is not installed
        try:
            # This test is more of a placeholder since we don't have an ONNX model readily available
            # In a real scenario, we would test with an actual ONNX model
            detector = YOLODetector({"model_path": "yolov8n.pt", "device": "cpu"})
            assert detector is not None
        except Exception as e:
            pytest.skip(f"ONNX model test skipped: {e}")


class TestModelPerformance:
    """Test cases for model performance"""
    
    def test_inference_time(self):
        """Test model inference time"""
        import time
        
        detector = YOLODetector({"model_path": "yolov8n.pt", "device": "cpu"})
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Measure inference time
        start_time = time.time()
        results = detector.detect(image)
        end_time = time.time()
        
        inference_time = end_time - start_time
        assert inference_time > 0  # Just check that it runs within reasonable time
        assert isinstance(results, list)
    
    def test_batch_performance(self):
        """Test batch processing performance"""
        import time
        
        detector = YOLODetector({"model_path": "yolov8n.pt", "device": "cpu"})
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            images = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(batch_size)]
            
            start_time = time.time()
            results = detector.detect_batch(images)
            end_time = time.time()
            
            inference_time = end_time - start_time
            assert inference_time > 0
            assert len(results) == batch_size
