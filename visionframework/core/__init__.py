"""
Core modules for vision framework
"""

from .base import BaseModule
from .detector import Detector
from .tracker import Tracker
from .pipeline import VisionPipeline
from .roi_detector import ROIDetector
from .counter import Counter

# Feature processors
from .processors import PoseEstimator, CLIPExtractor, ReIDExtractor, FeatureExtractor

# Data structures
from ..data import Detection, Track, STrack, Pose, KeyPoint, ROI

# Detector implementations
from .detectors import YOLODetector, DETRDetector

# Tracker implementations
from .trackers import IOUTracker, ByteTracker

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
    "IOUTracker",
    "ByteTracker",
]

