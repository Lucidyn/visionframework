"""
Vision Framework - A comprehensive framework for computer vision tasks
"""

# Core modules
from .core import (
    Detector, Tracker, VisionPipeline,
    ROIDetector, Counter, PoseEstimator
)

# Data structures
from .data import (
    Detection, Track, STrack, Pose, KeyPoint, ROI
)

# Detector implementations
from .core.detectors import YOLODetector, DETRDetector, RFDETRDetector

# Tracker implementations
from .core.trackers import IOUTracker, ByteTracker

# Utilities
from .utils import (
    Visualizer, Config, ImageUtils,
    ResultExporter, PerformanceMonitor, Timer,
    VideoProcessor, VideoWriter, process_video,
    TrajectoryAnalyzer, DetectionEvaluator, TrackingEvaluator
)

__version__ = "0.2.5"
__all__ = [
    # Core
    "Detector",
    "Tracker",
    "VisionPipeline",
    "ROIDetector",
    "Counter",
    "PoseEstimator",
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
    # Utilities
    "Visualizer",
    "Config",
    "ImageUtils",
    "ResultExporter",
    "PerformanceMonitor",
    "Timer",
    "VideoProcessor",
    "VideoWriter",
    "process_video",
    "TrajectoryAnalyzer",
    "DetectionEvaluator",
    "TrackingEvaluator",
]
