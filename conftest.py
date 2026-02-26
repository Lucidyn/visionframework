"""
Vision Framework 测试套件的共享 pytest fixtures。
"""

import numpy as np
import pytest

from visionframework import Detection, Track, Pose, KeyPoint


# ---------------------------------------------------------------------------
# 图像 fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_image() -> np.ndarray:
    """480×640 渐变 BGR 图像。"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, 640, dtype=np.uint8)
    img[:, :, 1] = np.linspace(0, 128, 640, dtype=np.uint8)
    return img


@pytest.fixture
def small_image() -> np.ndarray:
    """64×64 随机 BGR 图像。"""
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Detection fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_detection() -> Detection:
    """单个检测结果（person 类）。"""
    return Detection(
        bbox=(100.0, 150.0, 300.0, 400.0),
        confidence=0.85,
        class_id=0,
        class_name="person",
    )


@pytest.fixture
def dummy_detections() -> list:
    """包含 3 个检测结果的列表（2 person + 1 car）。"""
    return [
        Detection(bbox=(100, 150, 300, 400), confidence=0.85, class_id=0, class_name="person"),
        Detection(bbox=(350, 200, 500, 450), confidence=0.72, class_id=2, class_name="car"),
        Detection(bbox=(10,  10,  80,  80),  confidence=0.60, class_id=0, class_name="person"),
    ]


# ---------------------------------------------------------------------------
# Track fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_track() -> Track:
    """单个跟踪结果（ID=1，person 类）。"""
    return Track(
        track_id=1,
        bbox=(100.0, 150.0, 300.0, 400.0),
        confidence=0.85,
        class_id=0,
        class_name="person",
    )


@pytest.fixture
def dummy_tracks() -> list:
    """包含 2 个跟踪结果的列表。"""
    return [
        Track(track_id=1, bbox=(100, 150, 300, 400), confidence=0.85, class_id=0, class_name="person"),
        Track(track_id=2, bbox=(350, 200, 500, 450), confidence=0.72, class_id=2, class_name="car"),
    ]


# ---------------------------------------------------------------------------
# Pose fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_keypoints() -> list:
    """640×480 图像中心附近的 17 个 COCO 关键点。"""
    return [
        KeyPoint(x=320.0 + i * 5, y=240.0 + i * 3, confidence=0.9,
                 keypoint_id=i, keypoint_name=f"kp_{i}")
        for i in range(17)
    ]


@pytest.fixture
def dummy_pose(dummy_keypoints) -> Pose:
    """包含 17 个关键点的姿态结果。"""
    return Pose(
        bbox=(200.0, 100.0, 440.0, 460.0),
        confidence=0.88,
        keypoints=dummy_keypoints,
    )
