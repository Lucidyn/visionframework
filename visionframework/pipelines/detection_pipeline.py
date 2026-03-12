"""
Detection-only pipeline.
"""

from __future__ import annotations
from typing import Any, Dict

from visionframework.core.registry import PIPELINES
from .base import BasePipeline


@PIPELINES.register("detection")
class DetectionPipeline(BasePipeline):
    """Frame → Detector → detections list.

    Parameters
    ----------
    detector : Detector
        A detection algorithm instance.
    """

    def __init__(self, detector, **_kw):
        self.detector = detector

    def process(self, frame) -> Dict[str, Any]:
        detections = self.detector.predict(frame)
        return {"detections": detections}
