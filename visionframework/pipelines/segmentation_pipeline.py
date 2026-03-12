"""
Segmentation pipeline.
"""

from __future__ import annotations
from typing import Any, Dict

from visionframework.core.registry import PIPELINES
from .base import BasePipeline


@PIPELINES.register("segmentation")
class SegmentationPipeline(BasePipeline):
    """Frame → Segmenter → segmentation map.

    Parameters
    ----------
    segmenter : Segmenter
        A segmentation algorithm instance.
    """

    def __init__(self, segmenter, **_kw):
        self.segmenter = segmenter

    def process(self, frame) -> Dict[str, Any]:
        seg_map = self.segmenter.predict(frame)
        return {"seg_map": seg_map}
