"""
Detection + ReID + Tracking pipeline.
"""

from __future__ import annotations
from typing import Any, Dict

from visionframework.core.registry import PIPELINES
from .base import BasePipeline


@PIPELINES.register("reid_tracking")
class ReIDTrackingPipeline(BasePipeline):
    """Frame → Detector → ReID embedder → Tracker → tracks list.

    The embedder attaches appearance features that the tracker can use
    for more robust association (e.g. DeepSORT-style).

    Parameters
    ----------
    detector : Detector
    embedder : Embedder
    tracker : Any tracker from ``visionframework.algorithms.tracking`` implementing ``update(detections, frame)``
    """

    def __init__(self, detector, embedder, tracker, **_kw):
        self.detector = detector
        self.embedder = embedder
        self.tracker = tracker

    def process(self, frame) -> Dict[str, Any]:
        detections = self.detector.predict(frame)
        if detections:
            embeddings = self.embedder.extract(frame, detections)
            for det, emb in zip(detections, embeddings):
                det._embedding = emb
        tracks = self.tracker.update(detections, frame)
        return {"detections": detections, "tracks": tracks}

    def reset(self):
        self.tracker.reset()
