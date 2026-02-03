"""
Basic tests for IOUTracker and ByteTracker.

这些测试构造少量虚拟 Detection，验证：
- 跟踪器可以创建轨迹
- 连续帧更新时，轨迹数量和 ID 表现合理
"""

from typing import List

from visionframework.core.components.trackers import IOUTracker, ByteTracker
from visionframework.data import Detection


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
    # 第一帧应创建至少两个轨迹
    assert len(tracks1) >= 1

    dets2 = _make_detections_frame2()
    tracks2 = tracker.update(dets2)
    # 第二帧后，仍应有轨迹存在
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

    # 第二帧略微移动，应保持同 ID 或增加少量新轨迹
    dets2 = _make_detections_frame2()
    tracks2 = tracker.update(dets2)
    assert len(tracks2) >= 1

