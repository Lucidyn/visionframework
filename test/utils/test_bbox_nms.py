"""Tests for bbox and NMS utilities."""

import numpy as np
from visionframework.utils.bbox import xyxy2xywh, xywh2xyxy, clip_boxes
from visionframework.utils.nms import non_max_suppression


class TestBboxConversions:
    def test_xyxy_to_xywh(self):
        boxes = np.array([[10, 20, 30, 40]])
        result = xyxy2xywh(boxes)
        assert np.allclose(result, [[20, 30, 20, 20]])

    def test_xywh_to_xyxy(self):
        boxes = np.array([[20, 30, 20, 20]])
        result = xywh2xyxy(boxes)
        assert np.allclose(result, [[10, 20, 30, 40]])

    def test_roundtrip(self):
        original = np.array([[10, 20, 100, 200], [50, 60, 150, 250]])
        result = xywh2xyxy(xyxy2xywh(original))
        assert np.allclose(result, original)

    def test_clip_boxes(self):
        boxes = np.array([[-10, -5, 700, 500]])
        clipped = clip_boxes(boxes, (480, 640))
        assert clipped[0, 0] == 0
        assert clipped[0, 1] == 0
        assert clipped[0, 2] == 640
        assert clipped[0, 3] == 480


class TestNMS:
    def test_no_suppression(self):
        boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]])
        scores = np.array([0.9, 0.8])
        keep = non_max_suppression(boxes, scores, 0.5)
        assert len(keep) == 2

    def test_suppresses_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]])
        scores = np.array([0.9, 0.8])
        keep = non_max_suppression(boxes, scores, 0.3)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_empty_input(self):
        keep = non_max_suppression(np.empty((0, 4)), np.empty(0), 0.5)
        assert keep == []
