"""
Segmentation pipeline (YOLO 实例分割).
"""

from __future__ import annotations
from typing import Any, Dict

from visionframework.core.registry import PIPELINES
from .base import BasePipeline


@PIPELINES.register("segmentation")
class SegmentationPipeline(BasePipeline):
    """Frame → YOLO Segment → ``detections``（含 ``mask``）。"""

    def __init__(self, segmenter, **_kw):
        self.segmenter = segmenter

    def process(self, frame) -> Dict[str, Any]:
        detections = self.segmenter.predict(frame)
        return {"detections": detections}
