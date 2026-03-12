"""Tests for data structures."""

import numpy as np
from visionframework.data import Detection, Track, STrack, Pose, KeyPoint, ROI


class TestDetection:
    def test_creation(self):
        d = Detection(bbox=(10, 20, 30, 40), confidence=0.9, class_id=0, class_name="person")
        assert d.bbox == (10, 20, 30, 40)
        assert d.confidence == 0.9
        assert d.class_name == "person"

    def test_score_alias(self):
        d = Detection(bbox=(0, 0, 1, 1), confidence=0.5, class_id=0)
        assert d.score == 0.5
        d.score = 0.8
        assert d.confidence == 0.8

    def test_to_dict(self):
        d = Detection(bbox=(10, 20, 30, 40), confidence=0.9, class_id=0)
        data = d.to_dict()
        assert "bbox" in data
        assert "confidence" in data

    def test_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        d = Detection(bbox=(0, 0, 100, 100), confidence=0.9, class_id=0, mask=mask)
        assert d.has_mask()


class TestTrack:
    def test_update(self):
        t = Track(track_id=1, bbox=(10, 10, 50, 50), confidence=0.8, class_id=0)
        t.update((15, 15, 55, 55), 0.9)
        assert t.bbox == (15, 15, 55, 55)
        assert len(t.history) == 2

    def test_predict(self):
        t = Track(track_id=1, bbox=(10, 10, 50, 50), confidence=0.8, class_id=0)
        t.predict()
        assert t.age == 1
        assert t.time_since_update == 1


class TestSTrack:
    def test_activate(self):
        st = STrack(track_id=1, bbox=(10, 10, 50, 50), score=0.9, class_id=0)
        st.activate(frame_id=1)
        assert st.is_activated
        assert st.state == "Tracked"

    def test_mark_lost(self):
        st = STrack(track_id=1, bbox=(10, 10, 50, 50), score=0.9, class_id=0)
        st.activate(1)
        st.mark_lost()
        assert st.state == "Lost"

    def test_to_track(self):
        st = STrack(track_id=5, bbox=(10, 10, 50, 50), score=0.9, class_id=0, class_name="car")
        t = st.to_track()
        assert t.track_id == 5
        assert t.class_name == "car"


class TestROI:
    def test_rectangle_contains(self):
        roi = ROI(name="zone", points=[(0, 0), (100, 100)], roi_type="rectangle")
        assert roi.contains_point((50, 50))
        assert not roi.contains_point((200, 200))

    def test_bbox_center(self):
        roi = ROI(name="zone", points=[(0, 0), (100, 100)], roi_type="rectangle")
        assert roi.contains_bbox((10, 10, 90, 90))
        assert not roi.contains_bbox((200, 200, 300, 300))
