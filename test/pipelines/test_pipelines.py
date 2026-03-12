"""Tests for pipeline modules."""

import numpy as np
import torch
import pytest

from visionframework.core.builder import build_model
from visionframework.algorithms.detection.detector import Detector
from visionframework.algorithms.tracking.byte_tracker import ByteTracker
from visionframework.algorithms.tracking.iou_tracker import IOUTracker
from visionframework.pipelines.detection_pipeline import DetectionPipeline
from visionframework.pipelines.tracking_pipeline import TrackingPipeline
import visionframework.models  # noqa: F401


def _make_detector():
    cfg = {
        "backbone": {"type": "CSPDarknet", "depth": 0.33, "width": 0.25},
        "neck": {"type": "PAN", "in_channels": [64, 128, 256], "depth": 0.33},
        "head": {"type": "YOLOHead", "in_channels": [64, 128, 256], "num_classes": 80},
    }
    model = build_model(cfg)
    return Detector(model=model, device="cpu", conf=0.001, nms_iou=0.45)


class TestDetectionPipeline:
    def test_process_returns_dict(self):
        det = _make_detector()
        pipe = DetectionPipeline(detector=det)
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = pipe.process(frame)
        assert "detections" in result
        assert isinstance(result["detections"], list)


class TestTrackingPipeline:
    def test_process_returns_detections_and_tracks(self):
        det = _make_detector()
        tracker = ByteTracker(track_thresh=0.001)
        pipe = TrackingPipeline(detector=det, tracker=tracker)
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = pipe.process(frame)
        assert "detections" in result
        assert "tracks" in result

    def test_reset_clears_tracker(self):
        det = _make_detector()
        tracker = ByteTracker()
        pipe = TrackingPipeline(detector=det, tracker=tracker)
        pipe.process(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        pipe.reset()
        assert tracker.frame_id == 0
