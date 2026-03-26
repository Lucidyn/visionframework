"""Tests for tracker algorithms."""

import numpy as np
import pytest
from visionframework.data.detection import Detection
from visionframework.algorithms.tracking.byte_tracker import ByteTracker
from visionframework.algorithms.tracking.centroid_tracker import CentroidTracker
from visionframework.algorithms.tracking.deepsort_tracker import DeepSortTracker
from visionframework.algorithms.tracking.iou_tracker import IOUTracker
from visionframework.algorithms.tracking.sort_tracker import SortTracker
from visionframework.algorithms.tracking.utils import calculate_iou, iou_cost_matrix


class TestCalculateIoU:
    def test_perfect_overlap(self):
        assert calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_no_overlap(self):
        assert calculate_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0

    def test_partial_overlap(self):
        iou = calculate_iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert 0.1 < iou < 0.2

    def test_cost_matrix_shape(self):
        a = [(0, 0, 10, 10), (20, 20, 30, 30)]
        b = [(5, 5, 15, 15)]
        mat = iou_cost_matrix(a, b)
        assert mat.shape == (2, 1)


class TestByteTracker:
    def test_creates_tracks(self):
        tracker = ByteTracker(track_thresh=0.3)
        dets = [
            Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0),
            Detection(bbox=(300, 300, 400, 400), confidence=0.8, class_id=0),
        ]
        tracks = tracker.update(dets)
        assert len(tracks) == 2

    def test_maintains_track_ids(self):
        tracker = ByteTracker(track_thresh=0.3)
        dets = [Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)]
        tracks1 = tracker.update(dets)
        dets2 = [Detection(bbox=(105, 105, 205, 205), confidence=0.9, class_id=0)]
        tracks2 = tracker.update(dets2)
        assert tracks1[0].track_id == tracks2[0].track_id

    def test_empty_detections(self):
        tracker = ByteTracker(track_thresh=0.3)
        tracks = tracker.update([])
        assert tracks == []

    def test_none_detections(self):
        tracker = ByteTracker()
        tracks = tracker.update(None)
        assert tracks == []

    def test_reset(self):
        tracker = ByteTracker()
        tracker.update([Detection(bbox=(0, 0, 10, 10), confidence=0.9, class_id=0)])
        tracker.reset()
        assert tracker.frame_id == 0
        assert tracker.tracked_tracks == []


class TestIOUTracker:
    def test_creates_tracks(self):
        tracker = IOUTracker(min_hits=0)
        dets = [Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)]
        tracks = tracker.update(dets)
        assert len(tracks) == 1

    def test_maintains_ids(self):
        tracker = IOUTracker(min_hits=0)
        d1 = [Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)]
        t1 = tracker.update(d1)
        d2 = [Detection(bbox=(105, 105, 205, 205), confidence=0.9, class_id=0)]
        t2 = tracker.update(d2)
        assert t1[0].track_id == t2[0].track_id

    def test_empty_detections(self):
        tracker = IOUTracker(min_hits=1)
        tracks = tracker.update([])
        assert tracks == []

    def test_reset(self):
        tracker = IOUTracker()
        tracker.update([Detection(bbox=(0, 0, 10, 10), confidence=0.9, class_id=0)])
        tracker.reset()
        assert tracker.tracks == []


class TestCentroidTracker:
    def test_creates_tracks(self):
        tracker = CentroidTracker(min_hits=0, max_distance=200.0)
        dets = [Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)]
        tracks = tracker.update(dets)
        assert len(tracks) == 1

    def test_maintains_ids(self):
        tracker = CentroidTracker(min_hits=0, max_distance=200.0)
        t1 = tracker.update(
            [Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)]
        )
        t2 = tracker.update(
            [Detection(bbox=(105, 105, 205, 205), confidence=0.9, class_id=0)]
        )
        assert t1[0].track_id == t2[0].track_id


class TestSortTracker:
    def test_creates_tracks(self):
        tracker = SortTracker(min_hits=0)
        dets = [Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)]
        tracks = tracker.update(dets)
        assert len(tracks) == 1

    def test_maintains_ids(self):
        tracker = SortTracker(min_hits=0)
        t1 = tracker.update(
            [Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)]
        )
        t2 = tracker.update(
            [Detection(bbox=(105, 105, 205, 205), confidence=0.9, class_id=0)]
        )
        assert t1[0].track_id == t2[0].track_id


class TestDeepSortTracker:
    def test_iou_fallback_without_embedding(self):
        tracker = DeepSortTracker(min_hits=0)
        dets = [Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)]
        tracks = tracker.update(dets)
        assert len(tracks) == 1

    def test_embedding_association(self):
        tracker = DeepSortTracker(min_hits=0, lambda_emb=0.7)
        d1 = Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0)
        d1._embedding = np.ones(8, dtype=np.float32)
        d2 = Detection(bbox=(102, 102, 202, 202), confidence=0.9, class_id=0)
        d2._embedding = np.ones(8, dtype=np.float32)
        t1 = tracker.update([d1])
        t2 = tracker.update([d2])
        assert t1[0].track_id == t2[0].track_id
