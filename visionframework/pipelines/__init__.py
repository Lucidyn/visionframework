"""
Pipeline modules — orchestrate algorithms into task-level flows.
"""

from .base import BasePipeline
from .detection_pipeline import DetectionPipeline
from .segmentation_pipeline import SegmentationPipeline
from .tracking_pipeline import TrackingPipeline
from .reid_tracking_pipeline import ReIDTrackingPipeline

__all__ = [
    "BasePipeline",
    "DetectionPipeline",
    "SegmentationPipeline",
    "TrackingPipeline",
    "ReIDTrackingPipeline",
]
