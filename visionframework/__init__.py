"""
Vision Framework - A comprehensive framework for computer vision tasks
with lazy loading for heavy dependencies
"""

# Core modules - import base classes first
from .core import (
    Detector, Tracker, VisionPipeline,
    ROIDetector, Counter
)

# Data structures
from .data import (
    Detection, Track, STrack, Pose, KeyPoint, ROI
)

# Tracker implementations
from .core.trackers import IOUTracker, ByteTracker

# Exceptions
from .exceptions import (
    VisionFrameworkError,
    DetectorInitializationError,
    DetectorInferenceError,
    TrackerInitializationError,
    TrackerUpdateError,
    ConfigurationError,
    ModelNotFoundError,
    ModelLoadError,
    DeviceError,
    DependencyError,
    DataFormatError,
    ProcessingError
)

# Model management
from .models import ModelManager, get_model_manager

# Utilities - import lightweight utilities
from .utils import (
    Config, ImageUtils,
    ResultExporter, PerformanceMonitor, Timer,
    VideoProcessor, VideoWriter, process_video,
    TrajectoryAnalyzer
)


def __getattr__(name):
    """Lazy load heavy implementations to avoid importing heavy libraries at module load time"""
    # Detector implementations
    if name == "YOLODetector":
        from .core.detectors import YOLODetector
        return YOLODetector
    elif name == "DETRDetector":
        from .core.detectors import DETRDetector
        return DETRDetector
    elif name == "RFDETRDetector":
        from .core.detectors import RFDETRDetector
        return RFDETRDetector
    # Heavy processors
    elif name == "PoseEstimator":
        from .core import PoseEstimator
        return PoseEstimator
    elif name == "CLIPExtractor":
        from .core import CLIPExtractor
        return CLIPExtractor
    elif name == "ReIDExtractor":
        from .core import ReIDExtractor
        return ReIDExtractor
    # Heavy utilities
    elif name == "Visualizer":
        from .utils import Visualizer
        return Visualizer
    elif name == "DetectionEvaluator":
        from .utils import DetectionEvaluator
        return DetectionEvaluator
    elif name == "TrackingEvaluator":
        from .utils import TrackingEvaluator
        return TrackingEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__version__ = "0.2.8"
__all__ = [
    # Core
    "Detector",
    "Tracker",
    "VisionPipeline",
    "ROIDetector",
    "Counter",
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
    # Exceptions
    "VisionFrameworkError",
    "DetectorInitializationError",
    "DetectorInferenceError",
    "TrackerInitializationError",
    "TrackerUpdateError",
    "ConfigurationError",
    "ModelNotFoundError",
    "ModelLoadError",
    "DeviceError",
    "DependencyError",
    "DataFormatError",
    "ProcessingError",
    # Model management
    "ModelManager",
    "get_model_manager",
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

