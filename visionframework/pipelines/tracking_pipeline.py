"""
Detection + Tracking pipeline.
"""

from __future__ import annotations
from typing import Any, Dict

from visionframework.core.registry import PIPELINES
from .base import BasePipeline


@PIPELINES.register("tracking")
class TrackingPipeline(BasePipeline):
    """Frame → Detector → Tracker → tracks list.

    Parameters
    ----------
    detector : Detector
        Detection algorithm.
    tracker : ByteTracker | IOUTracker
        Tracking algorithm.
    """

    def __init__(self, detector, tracker, **_kw):
        self.detector = detector
        self.tracker = tracker

    def process(self, frame) -> Dict[str, Any]:
        detections = self.detector.predict(frame)
        tracks = self.tracker.update(detections, frame)
        return {"detections": detections, "tracks": tracks}

    def reset(self):
        self.tracker.reset()
