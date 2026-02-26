"""
ROI 区域检测器与计数器测试。
"""

import pytest
import numpy as np

from visionframework import (
    ROIDetector,
    Counter,
    Detection,
    Track,
    ROI,
)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_detection(cx: float, cy: float, w: float = 40, h: float = 40,
                    class_name: str = "person") -> Detection:
    return Detection(
        bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        confidence=0.9,
        class_id=0,
        class_name=class_name,
    )


def _make_track(track_id: int, cx: float, cy: float,
                w: float = 40, h: float = 40) -> Track:
    return Track(
        track_id=track_id,
        bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        confidence=0.9,
        class_id=0,
        class_name="person",
    )


# 正方形 ROI：(100,100) → (400,400)
_SQUARE_POINTS = [(100, 100), (400, 100), (400, 400), (100, 400)]


# ---------------------------------------------------------------------------
# ROI 数据结构
# ---------------------------------------------------------------------------

def test_roi_creation():
    roi = ROI("zone1", _SQUARE_POINTS)
    assert roi.name == "zone1"
    assert roi.type == "polygon"


def test_roi_contains_point_inside():
    roi = ROI("zone1", _SQUARE_POINTS)
    assert roi.contains_point((250, 250)) is True


def test_roi_contains_point_outside():
    roi = ROI("zone1", _SQUARE_POINTS)
    assert roi.contains_point((50, 50)) is False


def test_roi_contains_bbox_center_inside():
    roi = ROI("zone1", _SQUARE_POINTS)
    assert roi.contains_bbox((230, 230, 270, 270)) is True


def test_roi_contains_bbox_center_outside():
    roi = ROI("zone1", _SQUARE_POINTS)
    assert roi.contains_bbox((30, 30, 70, 70)) is False


def test_roi_rectangle_type():
    roi = ROI("rect", [(100, 100), (400, 400)], roi_type="rectangle")
    assert roi.contains_point((250, 250)) is True
    assert roi.contains_point((50, 50)) is False


def test_roi_get_mask():
    roi = ROI("zone1", _SQUARE_POINTS)
    mask = roi.get_mask((480, 640))
    assert mask.shape == (480, 640)
    assert mask[250, 250] == 255
    assert mask[50, 50] == 0


# ---------------------------------------------------------------------------
# ROIDetector
# ---------------------------------------------------------------------------

def test_roi_detector_creation():
    det = ROIDetector()
    assert isinstance(det, ROIDetector)


def test_roi_detector_initialize():
    det = ROIDetector()
    result = det.initialize()
    assert result is True
    assert det.is_initialized is True


def test_roi_detector_add_roi():
    det = ROIDetector()
    det.add_roi("zone1", _SQUARE_POINTS)
    rois = det.get_rois()
    assert len(rois) == 1
    assert rois[0].name == "zone1"


def test_roi_detector_get_roi_by_name():
    det = ROIDetector()
    det.add_roi("zone1", _SQUARE_POINTS)
    roi = det.get_roi_by_name("zone1")
    assert roi is not None
    assert roi.name == "zone1"
    assert det.get_roi_by_name("nonexistent") is None


def test_roi_detector_check_detection_in_roi():
    det = ROIDetector()
    det.add_roi("zone1", _SQUARE_POINTS)

    inside_det = _make_detection(250, 250)
    outside_det = _make_detection(50, 50)

    assert det.check_detection_in_roi(inside_det, "zone1") is True
    assert det.check_detection_in_roi(outside_det, "zone1") is False


def test_roi_detector_check_track_in_roi():
    det = ROIDetector()
    det.add_roi("zone1", _SQUARE_POINTS)

    inside_track = _make_track(1, 250, 250)
    outside_track = _make_track(2, 50, 50)

    assert det.check_track_in_roi(inside_track, "zone1") is True
    assert det.check_track_in_roi(outside_track, "zone1") is False


def test_roi_detector_filter_detections_by_roi():
    det = ROIDetector()
    det.add_roi("zone1", _SQUARE_POINTS)

    detections = [
        _make_detection(250, 250),
        _make_detection(50, 50),
        _make_detection(300, 300),
    ]
    filtered = det.filter_detections_by_roi(detections, "zone1")
    assert len(filtered) == 2


def test_roi_detector_filter_tracks_by_roi():
    det = ROIDetector()
    det.add_roi("zone1", _SQUARE_POINTS)

    tracks = [
        _make_track(1, 250, 250),
        _make_track(2, 50, 50),
    ]
    filtered = det.filter_tracks_by_roi(tracks, "zone1")
    assert len(filtered) == 1
    assert filtered[0].track_id == 1


def test_roi_detector_get_detections_by_roi():
    det = ROIDetector()
    det.add_roi("zone1", _SQUARE_POINTS)

    detections = [
        _make_detection(250, 250),
        _make_detection(50, 50),
    ]
    grouped = det.get_detections_by_roi(detections)
    assert "zone1" in grouped
    assert "none" in grouped
    assert len(grouped["zone1"]) == 1
    assert len(grouped["none"]) == 1


def test_roi_detector_process():
    det = ROIDetector()
    det.add_roi("zone1", _SQUARE_POINTS)

    detections = [_make_detection(250, 250), _make_detection(50, 50)]
    result = det.process(detections, "zone1")
    assert len(result) == 1


def test_roi_detector_process_batch():
    det = ROIDetector()
    det.add_roi("zone1", _SQUARE_POINTS)

    batch = [
        [_make_detection(250, 250), _make_detection(50, 50)],
        [_make_detection(300, 300)],
    ]
    results = det.process_batch(batch, "zone1")
    assert len(results) == 2
    assert len(results[0]) == 1
    assert len(results[1]) == 1


def test_roi_detector_from_config():
    config = {
        "rois": [
            {"name": "entrance", "points": _SQUARE_POINTS, "type": "polygon"}
        ]
    }
    det = ROIDetector(config)
    rois = det.get_rois()
    assert len(rois) == 1
    assert rois[0].name == "entrance"


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------

def test_counter_creation():
    counter = Counter()
    assert isinstance(counter, Counter)


def test_counter_initialize():
    counter = Counter()
    result = counter.initialize()
    assert result is True


def test_counter_count_detections_no_roi():
    """没有 ROI 时，Counter 返回空字典。"""
    counter = Counter()
    counter.initialize()
    detections = [_make_detection(250, 250)]
    result = counter.count_detections(detections)
    assert isinstance(result, dict)


def test_counter_count_detections_with_roi():
    roi_detector = ROIDetector()
    roi_detector.add_roi("zone1", _SQUARE_POINTS)
    roi_detector.initialize()

    counter = Counter({"roi_detector": roi_detector})
    counter.initialize()

    detections = [_make_detection(250, 250), _make_detection(50, 50)]
    result = counter.count_detections(detections, "zone1")
    assert "zone1" in result
    assert result["zone1"]["count"] == 1


def test_counter_count_tracks():
    roi_detector = ROIDetector()
    roi_detector.add_roi("zone1", _SQUARE_POINTS)
    roi_detector.initialize()

    counter = Counter({"roi_detector": roi_detector})
    counter.initialize()

    tracks = [
        _make_track(1, 250, 250),
        _make_track(2, 50, 50),
    ]
    result = counter.count_tracks(tracks, "zone1")
    assert "zone1" in result
    assert result["zone1"]["inside"] == 1


def test_counter_entering_exiting():
    """目标进出 ROI 时计数正确。"""
    roi_detector = ROIDetector()
    roi_detector.add_roi("zone1", _SQUARE_POINTS)
    roi_detector.initialize()

    counter = Counter({"roi_detector": roi_detector})
    counter.initialize()

    # 帧1：track1 在外，track2 在内
    tracks_f1 = [_make_track(1, 50, 50), _make_track(2, 250, 250)]
    counter.count_tracks(tracks_f1, "zone1")

    # 帧2：track1 进入，track2 离开
    tracks_f2 = [_make_track(1, 250, 250), _make_track(2, 50, 50)]
    result = counter.count_tracks(tracks_f2, "zone1")

    assert result["zone1"]["entering"] == 1
    assert result["zone1"]["exiting"] == 1


def test_counter_get_counts():
    roi_detector = ROIDetector()
    roi_detector.add_roi("zone1", _SQUARE_POINTS)
    roi_detector.initialize()

    counter = Counter({"roi_detector": roi_detector})
    counter.initialize()

    tracks = [_make_track(1, 250, 250)]
    counter.count_tracks(tracks, "zone1")

    counts = counter.get_counts("zone1")
    assert isinstance(counts, dict)


def test_counter_reset_counts():
    roi_detector = ROIDetector()
    roi_detector.add_roi("zone1", _SQUARE_POINTS)
    roi_detector.initialize()

    counter = Counter({"roi_detector": roi_detector})
    counter.initialize()

    tracks = [_make_track(1, 250, 250)]
    counter.count_tracks(tracks, "zone1")
    counter.reset_counts("zone1")
    counts = counter.get_counts("zone1")
    assert counts.get("total", 0) == 0


def test_counter_process_alias():
    roi_detector = ROIDetector()
    roi_detector.add_roi("zone1", _SQUARE_POINTS)
    roi_detector.initialize()

    counter = Counter({"roi_detector": roi_detector})
    counter.initialize()

    tracks = [_make_track(1, 250, 250)]
    result = counter.process(tracks, "zone1")
    assert isinstance(result, dict)
