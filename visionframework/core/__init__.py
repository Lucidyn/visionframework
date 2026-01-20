"""
Core modules for vision framework with lazy loading for heavy dependencies
"""

from .base import BaseModule
from .detector import Detector
from .tracker import Tracker
from .pipeline import VisionPipeline
from .roi_detector import ROIDetector
from .counter import Counter

# Feature processors (lazy load processors that import heavy libs)
from .processors import FeatureExtractor

# Data structures
from ..data import Detection, Track, STrack, Pose, KeyPoint, ROI


def __getattr__(name):
    """Lazy load detector, tracker and processor implementations to avoid heavy imports at module load time"""
    # Lazy load detectors
    if name == "YOLODetector":
        from .detectors import YOLODetector
        return YOLODetector
    elif name == "DETRDetector":
        from .detectors import DETRDetector
        return DETRDetector
    elif name == "RFDETRDetector":
        from .detectors import RFDETRDetector
        return RFDETRDetector
    # Lazy load trackers
    elif name == "IOUTracker":
        from .trackers import IOUTracker
        return IOUTracker
    elif name == "ByteTracker":
        from .trackers import ByteTracker
        return ByteTracker
    elif name == "ReIDTracker":
        from .trackers import ReIDTracker
        return ReIDTracker
    # Lazy load heavy processors
    elif name == "PoseEstimator":
        from .processors import PoseEstimator
        return PoseEstimator
    elif name == "CLIPExtractor":
        from .processors import CLIPExtractor
        return CLIPExtractor
    elif name == "ReIDExtractor":
        from .processors import ReIDExtractor
        return ReIDExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseModule",
    "Detector",
    "Tracker",
    "VisionPipeline",
    "ROIDetector",
    "Counter",
    # Feature processors
    "FeatureExtractor",
    "PoseEstimator",
    "CLIPExtractor",
    "ReIDExtractor",
    # Data structures
    "Detection",
    "Track",
    "STrack",
    "Pose",
    "KeyPoint",
    "ROI",
    # Implementations
    "YOLODetector",
    "DETRDetector",
    "RFDETRDetector",
    "IOUTracker",
    "ByteTracker",
]

