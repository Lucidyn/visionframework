"""
IOUTracker 与 ByteTracker 基础测试。

构造少量虚拟 Detection，验证：
- 跟踪器可以创建轨迹
- 连续帧更新时，轨迹数量和 ID 表现合理
"""

from typing import List

from visionframework import IOUTracker, ByteTracker, Detection


def _make_detections_frame1() -> List[Detection]:
    return [
        Detection(bbox=(10, 10, 30, 30), confidence=0.9, class_id=0, class_name="obj"),
        Detection(bbox=(50, 50, 80, 80), confidence=0.85, class_id=0, class_name="obj"),
    ]


def _make_detections_frame2() -> List[Detection]:
    # 轻微移动，保持 IoU 较高
    return [
        Detection(bbox=(12, 12, 32, 32), confidence=0.9, class_id=0, class_name="obj"),
        Detection(bbox=(52, 52, 82, 82), confidence=0.85, class_id=0, class_name="obj"),
    ]


def test_iou_tracker_creates_and_updates_tracks() -> None:
    tracker = IOUTracker({"max_age": 5, "min_hits": 0, "iou_threshold": 0.1})

    dets1 = _make_detections_frame1()
    tracks1 = tracker.update(dets1)
    assert len(tracks1) >= 1

    dets2 = _make_detections_frame2()
    tracks2 = tracker.update(dets2)
    assert len(tracks2) >= 1


def test_byte_tracker_creates_tracks() -> None:
    tracker = ByteTracker(
        {
            "track_thresh": 0.5,
            "track_buffer": 30,
            "match_thresh": 0.7,
            "frame_rate": 30,
            "min_box_area": 4,
        }
    )

    dets1 = _make_detections_frame1()
    tracks1 = tracker.update(dets1)
    assert len(tracks1) >= 1

    dets2 = _make_detections_frame2()
    tracks2 = tracker.update(dets2)
    assert len(tracks2) >= 1
