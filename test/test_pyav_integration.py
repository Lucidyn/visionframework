"""
Test PyAV integration for video processing

This module tests the PyAV integration for high-performance video processing,
including PyAVVideoProcessor, PyAVVideoWriter, and VisionPipeline integration.
"""

import cv2
import numpy as np
import os
import tempfile
from unittest import TestCase, skipIf
from visionframework.utils.io import PyAVVideoProcessor, PyAVVideoWriter, process_video
from visionframework import VisionPipeline

# Try to import pyav for conditional testing
try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False


class TestPyAVIntegration(TestCase):
    """Test PyAV integration functionality"""
    
    def setUp(self):
        """Set up test resources"""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test video file for testing
        self.test_video_path = os.path.join(self.temp_dir, "test_video.mp4")
        self._create_test_video()
    
    def tearDown(self):
        """Clean up test resources"""
        # Remove temporary files
        if os.path.exists(self.test_video_path):
            os.remove(self.test_video_path)
        
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def _create_test_video(self):
        """Create a simple test video file"""
        # Create a 10-second test video with moving square
        height, width = 480, 640
        fps = 30
        total_frames = 300
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(self.test_video_path, fourcc, fps, (width, height))
        
        for i in range(total_frames):
            # Create black frame
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Draw moving square
            square_size = 50
            x = int((width - square_size) * (i / total_frames))
            y = int(height / 2 - square_size / 2)
            cv2.rectangle(frame, (x, y), (x + square_size, y + square_size), (0, 255, 0), -1)
            
            # Draw frame number
            cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            writer.write(frame)
        
        writer.release()
    
    @skipIf(not PYAV_AVAILABLE, "PyAV not installed")
    def test_pyav_video_processor(self):
        """Test PyAVVideoProcessor functionality"""
        # Test initialization
        processor = PyAVVideoProcessor(self.test_video_path)
        self.assertTrue(processor is not None)
        
        # Test opening video
        self.assertTrue(processor.open())
        
        # Test getting video info
        info = processor.get_info()
        self.assertIn("fps", info)
        self.assertIn("width", info)
        self.assertIn("height", info)
        self.assertIn("total_frames", info)
        self.assertEqual(info["backend"], "pyav")
        
        # Test reading frames
        frames_read = 0
        while True:
            ret, frame = processor.read_frame()
            if not ret:
                break
            self.assertTrue(frame is not None)
            self.assertEqual(len(frame.shape), 3)  # Should be (H, W, 3)
            frames_read += 1
        
        self.assertGreater(frames_read, 0)
        
        # Test closing
        processor.close()
    
    @skipIf(not PYAV_AVAILABLE, "PyAV not installed")
    def test_pyav_video_writer(self):
        """Test PyAVVideoWriter functionality"""
        output_path = os.path.join(self.temp_dir, "test_output.mp4")
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Test Frame", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Test initialization and writing
        with PyAVVideoWriter(output_path, fps=30.0, frame_size=(640, 480)) as writer:
            # Test writing multiple frames
            for i in range(10):
                # Modify frame for each iteration
                frame_copy = frame.copy()
                cv2.putText(frame_copy, f"Frame: {i}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.assertTrue(writer.write(frame_copy))
        
        # Test that output file was created
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
    
    @skipIf(not PYAV_AVAILABLE, "PyAV not installed")
    def test_process_video_with_pyav(self):
        """Test process_video function with PyAV backend"""
        output_path = os.path.join(self.temp_dir, "processed_output.mp4")
        
        # Test frame callback
        def frame_callback(frame, frame_number):
            # Draw frame number on frame
            cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame
        
        # Test processing with PyAV
        success = process_video(
            input_path=self.test_video_path,
            output_path=output_path,
            frame_callback=frame_callback,
            use_pyav=True
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
    
    @skipIf(not PYAV_AVAILABLE, "PyAV not installed")
    def test_vision_pipeline_with_pyav(self):
        """Test VisionPipeline with PyAV backend"""
        output_path = os.path.join(self.temp_dir, "pipeline_output.mp4")
        
        # Create pipeline
        pipeline = VisionPipeline({
            "enable_tracking": True,
            "detector_config": {
                "model_type": "yolo",
                "model_path": "yolov8n.pt",
                "conf_threshold": 0.25
            }
        })
        
        # Initialize pipeline
        pipeline.initialize()
        
        # Test video processing with PyAV
        success = pipeline.process_video(
            input_source=self.test_video_path,
            output_path=output_path,
            use_pyav=True
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
        
        pipeline.cleanup()
    
    @skipIf(not PYAV_AVAILABLE, "PyAV not installed")
    def test_vision_pipeline_run_video_with_pyav(self):
        """Test VisionPipeline.run_video with PyAV backend"""
        output_path = os.path.join(self.temp_dir, "run_video_output.mp4")
        
        # Test run_video with PyAV
        success = VisionPipeline.run_video(
            input_source=self.test_video_path,
            output_path=output_path,
            model_path="yolov8n.pt",
            enable_tracking=True,
            use_pyav=True
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
    
    def test_process_video_fallback_to_opencv(self):
        """Test process_video falls back to OpenCV when PyAV is not available"""
        output_path = os.path.join(self.temp_dir, "fallback_output.mp4")
        
        # Test frame callback
        def frame_callback(frame, frame_number):
            cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return frame
        
        # Test processing with use_pyav=True but it should fallback if PyAV not available
        success = process_video(
            input_path=self.test_video_path,
            output_path=output_path,
            frame_callback=frame_callback,
            use_pyav=True  # This should work even if PyAV is not installed
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)
        
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
    
    @skipIf(not PYAV_AVAILABLE, "PyAV not installed")
    def test_pyav_camera_fallback(self):
        """Test PyAV falls back to OpenCV for camera inputs"""
        # Test that PyAV properly falls back for camera input
        # We'll test with camera index 999 (which shouldn't exist)
        # but the important part is that it tries to use OpenCV
        
        # Test with process_video
        success = process_video(
            input_path=999,  # Invalid camera
            use_pyav=True
        )
        # Should return False because camera doesn't exist, but should not crash
        self.assertFalse(success)
        
        # Test with VisionPipeline
        pipeline = VisionPipeline()
        pipeline.initialize()
        
        success = pipeline.process_video(
            input_source=999,  # Invalid camera
            use_pyav=True
        )
        # Should return False because camera doesn't exist, but should not crash
        self.assertFalse(success)
        
        pipeline.cleanup()
    
    @skipIf(not PYAV_AVAILABLE, "PyAV not installed")
    def test_pyav_stream_fallback(self):
        """Test PyAV falls back to OpenCV for stream inputs"""
        # Test that PyAV properly falls back for stream input
        
        # Test with process_video
        success = process_video(
            input_path="rtsp://example.com/stream",  # Invalid stream
            use_pyav=True
        )
        # Should return False because stream doesn't exist, but should not crash
        self.assertFalse(success)


if __name__ == "__main__":
    import unittest
    unittest.main()
