import numpy as np
import pytest
import cv2
from visionframework.core.trackers.byte_tracker import ByteTracker
from visionframework.core.trackers.iou_tracker import IOUTracker


class TestTrackers:
    """Test cases for different tracker implementations"""
    
    def test_bytetracker_initialization(self):
        """Test ByteTracker initialization"""
        tracker = ByteTracker()
        assert tracker is not None
        assert hasattr(tracker, "update")
    
    def test_ioutracker_initialization(self):
        """Test IOUTracker initialization"""
        tracker = IOUTracker()
        assert tracker is not None
        assert hasattr(tracker, "update")
    
    def test_tracker_update(self):
        """Test tracker update functionality"""
        tracker = IOUTracker()
        
        # Create dummy detections (using the actual Detection structure)
        from visionframework.data.detection import Detection
        detections = [
            Detection(
                bbox=(100, 100, 200, 200),
                confidence=0.9,
                class_id=0,
                class_name="person"
            )
        ]
        
        # Create dummy frame
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        
        results = tracker.update(detections, frame)
        assert isinstance(results, list)
    
    def test_tracker_reset(self):
        """Test tracker reset functionality"""
        tracker = IOUTracker()
        
        # Create dummy detections
        from visionframework.data.detection import Detection
        detections = [
            Detection(
                bbox=(100, 100, 200, 200),
                confidence=0.9,
                class_id=0,
                class_name="person"
            )
        ]
        
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Update tracker
        results = tracker.update(detections, frame)
        
        # Reset tracker
        tracker.reset()
