import os
import tempfile
import numpy as np
import pytest
import cv2
from visionframework.utils.config import Config
from visionframework.utils.image_utils import ImageUtils
from visionframework.utils.performance import PerformanceMonitor


class TestConfigUtils:
    """Test cases for configuration utilities"""
    
    def test_load_save_config(self):
        """Test loading and saving configuration"""
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "test_config.yaml")
            
            # Create a test config
            test_config = {
                "model": {
                    "path": "yolov8n.pt",
                    "device": "auto"
                },
                "detection": {
                    "conf_threshold": 0.25
                }
            }
            
            # Save the config
            Config.save_to_file(test_config, config_path)
            assert os.path.exists(config_path)
            
            # Load the config
            loaded_config = Config.load_from_file(config_path)
            assert isinstance(loaded_config, dict)
            assert loaded_config == test_config
    
    def test_load_nonexistent_config(self):
        """Test loading a nonexistent config file"""
        with pytest.raises(Exception):
            Config.load_from_file("nonexistent_config.yaml")


class TestImageUtils:
    """Test cases for image utilities"""
    
    def test_resize_image(self):
        """Test image resizing"""
        # Create a dummy image
        image = np.zeros((640, 480, 3), dtype=np.uint8)
        
        # Test resizing to specific dimensions
        resized = ImageUtils.resize_image(image, target_size=(320, 240), keep_aspect=False)
        assert resized.shape == (240, 320, 3)
        
        # Test resizing with keep_aspect=True
        resized_keep_ratio = ImageUtils.resize_image(image, target_size=(320, 320), keep_aspect=True)
        assert resized_keep_ratio.shape[0] <= 320
        assert resized_keep_ratio.shape[1] <= 320
    
    def test_load_save_image(self):
        """Test loading and saving images"""
        # Create a dummy image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (80, 80), (255, 255, 255), -1)
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = os.path.join(tmpdir, "test_image.jpg")
            
            # Save the image
            success = ImageUtils.save_image(image, image_path)
            assert success
            assert os.path.exists(image_path)
            
            # Load the image
            loaded_image = ImageUtils.load_image(image_path)
            assert loaded_image is not None
            assert loaded_image.shape == image.shape


class TestPerformanceUtils:
    """Test cases for performance utilities"""
    
    def test_performance_monitor(self):
        """Test PerformanceMonitor functionality"""
        monitor = PerformanceMonitor()
        
        # Test start method
        monitor.start()
        # Simulate some operation with multiple ticks
        import time
        for _ in range(3):
            monitor.tick()
            time.sleep(0.01)
        
        # Test getting metrics
        metrics = monitor.get_metrics()
        assert metrics.frame_count >= 3
        assert metrics.fps >= 0
        
        # Test reset
        monitor.reset()
        metrics_after_reset = monitor.get_metrics()
        assert metrics_after_reset.frame_count == 0
    
    def test_performance_monitor_basic(self):
        """Test basic PerformanceMonitor functionality"""
        monitor = PerformanceMonitor()
        monitor.start()
        
        # Simulate multiple frames
        for _ in range(5):
            monitor.tick()
            import time
            time.sleep(0.01)
        
        # Check if metrics were recorded
        metrics = monitor.get_metrics()
        assert metrics.frame_count == 5
        assert metrics.fps >= 0
        assert metrics.avg_fps >= 0
