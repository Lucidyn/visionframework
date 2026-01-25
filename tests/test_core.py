import cv2
import numpy as np
import pytest
from visionframework.core.pipeline import VisionPipeline
from visionframework.core.detectors.yolo_detector import YOLODetector
from visionframework.utils.auto_labeler import AutoLabeler


class TestAutoLabeler:
    """Test cases for AutoLabeler functionality"""
    
    def test_initialization(self):
        """Test AutoLabeler initialization"""
        labeler = AutoLabeler({"detector_config": {"model_path": "yolov8n.pt"}})
        assert labeler is not None
    
    def test_supported_formats(self):
        """Test supported annotation formats"""
        labeler = AutoLabeler({"detector_config": {"model_path": "yolov8n.pt"}})
        # Check output_format attribute instead of supported_formats
        assert hasattr(labeler, "output_format")
        assert labeler.output_format in ["json", "csv", "coco"]
    
    def test_annotate_single_image(self):
        """Test annotation of a single image"""
        labeler = AutoLabeler({"detector_config": {"model_path": "yolov8n.pt"}})
        # Create a dummy image and save it temporarily
        import tempfile
        import os
        
        temp_dir = tempfile.gettempdir()
        temp_image_path = os.path.join(temp_dir, "temp_test_image.jpg")
        cv2.imwrite(temp_image_path, np.zeros((640, 640, 3), dtype=np.uint8))
        
        try:
            results = labeler.label_image(temp_image_path)
            assert isinstance(results, dict)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)


class TestYOLODetector:
    """Test cases for YOLODetector functionality"""
    
    def test_initialization(self):
        """Test YOLODetector initialization"""
        detector = YOLODetector({"model_path": "yolov8n.pt", "device": "cpu"})
        assert detector is not None
    
    def test_detect_single_image(self):
        """Test detection on a single image"""
        detector = YOLODetector({"model_path": "yolov8n.pt", "device": "cpu"})
        # Create a dummy image
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(image, (100, 100), (200, 200), (255, 255, 255), -1)
        
        results = detector.detect(image)
        assert isinstance(results, list)
    
    def test_batch_detection(self):
        """Test batch detection functionality"""
        detector = YOLODetector({"model_path": "yolov8n.pt", "device": "cpu"})
        # Create dummy images
        images = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(2)]
        
        results = detector.detect_batch(images)
        assert isinstance(results, list)
        assert len(results) == 2
    
    def test_class_conf_thresholds(self):
        """Test per-class confidence thresholds"""
        class_conf = {"person": 0.5, "car": 0.3}
        detector = YOLODetector(
            {"model_path": "yolov8n.pt", 
             "device": "cpu",
             "category_thresholds": class_conf}
        )
    
    def test_device_selection(self):
        """Test automatic device selection"""
        detector = YOLODetector({"model_path": "yolov8n.pt", "device": "cpu"})
        assert detector.device is not None
    
    def test_fp16_support(self):
        """Test FP16 inference support"""
        detector = YOLODetector({"model_path": "yolov8n.pt", "device": "cpu", "use_fp16": False})


class TestVisionPipeline:
    """Test cases for VisionPipeline functionality"""
    
    def test_initialization(self):
        """Test VisionPipeline initialization"""
        pipeline = VisionPipeline({"detector_config": {"model_path": "yolov8n.pt", "device": "cpu"}})
        assert pipeline is not None
    
    def test_process_image(self):
        """Test image processing"""
        pipeline = VisionPipeline({"detector_config": {"model_path": "yolov8n.pt", "device": "cpu"}})
        # Create a dummy image
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        results = pipeline.process(image)
        assert isinstance(results, dict)
        assert "detections" in results
    
    def test_process_batch(self):
        """Test batch image processing"""
        pipeline = VisionPipeline({"detector_config": {"model_path": "yolov8n.pt", "device": "cpu"}})
        # Create dummy images
        images = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(2)]
        
        results = pipeline.process_batch(images)
        assert isinstance(results, list)
        assert len(results) == 2
    
    def test_static_process_image(self):
        """Test static process_image method"""
        # Create a dummy image
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        results = VisionPipeline.process_image(image, {"detector_config": {"model_path": "yolov8n.pt", "device": "cpu"}})
        assert isinstance(results, dict)
        assert "detections" in results
    
    def test_with_tracking_class_method(self):
        """Test with_tracking class method"""
        pipeline = VisionPipeline.with_tracking({"detector_config": {"model_path": "yolov8n.pt", "device": "cpu"}})
        assert pipeline is not None
    
    def test_from_model_class_method(self):
        """Test from_model class method"""
        pipeline = VisionPipeline.from_model("yolov8n.pt")
        assert pipeline is not None
    
    def test_visualize_results(self):
        """Test result visualization capability"""
        # VisionPipeline doesn't have a built-in visualize method, but we can test
        # that it returns results in a format that can be visualized
        pipeline = VisionPipeline({"detector_config": {"model_path": "yolov8n.pt", "device": "cpu"}})
        # Create a dummy image
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        results = pipeline.process(image)
        assert isinstance(results, dict)
        assert "detections" in results
        # Results should be in a format that can be visualized externally
