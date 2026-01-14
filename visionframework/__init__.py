"""
Vision Framework - A comprehensive framework for computer vision tasks
"""

# Core modules
from .core import (
    Detector, Tracker, VisionPipeline,
    ROIDetector, Counter, PoseEstimator, CLIPExtractor, ReIDExtractor
)

# Data structures
from .data import (
    Detection, Track, STrack, Pose, KeyPoint, ROI
)

# Detector implementations
from .core.detectors import YOLODetector, DETRDetector, RFDETRDetector

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

# Utilities
from .utils import (
    Visualizer, Config, ImageUtils,
    ResultExporter, PerformanceMonitor, Timer,
    VideoProcessor, VideoWriter, process_video,
    TrajectoryAnalyzer, DetectionEvaluator, TrackingEvaluator
)

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

