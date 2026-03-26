"""
Shared pytest fixtures.
"""

import os

# 默认安静，避免 TaskRunner 等在测试里输出 INFO；若需调试可设 VISIONFRAMEWORK_LOG_LEVEL=INFO
os.environ.setdefault("VISIONFRAMEWORK_LOG_LEVEL", "WARNING")

import numpy as np
import pytest

from visionframework.data import Detection, Track, STrack, Pose, KeyPoint


@pytest.fixture
def dummy_image() -> np.ndarray:
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def small_image() -> np.ndarray:
    return np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def dummy_detection() -> Detection:
    return Detection(
        bbox=(100.0, 150.0, 300.0, 400.0),
        confidence=0.85, class_id=0, class_name="person",
    )


@pytest.fixture
def dummy_detections() -> list:
    return [
        Detection(bbox=(100, 150, 300, 400), confidence=0.85, class_id=0, class_name="person"),
        Detection(bbox=(350, 200, 500, 450), confidence=0.72, class_id=2, class_name="car"),
        Detection(bbox=(10, 10, 80, 80), confidence=0.60, class_id=0, class_name="person"),
    ]


@pytest.fixture
def dummy_track() -> Track:
    return Track(
        track_id=1, bbox=(100.0, 150.0, 300.0, 400.0),
        confidence=0.85, class_id=0, class_name="person",
    )


@pytest.fixture
def dummy_tracks() -> list:
    return [
        Track(track_id=1, bbox=(100, 150, 300, 400), confidence=0.85, class_id=0, class_name="person"),
        Track(track_id=2, bbox=(350, 200, 500, 450), confidence=0.72, class_id=2, class_name="car"),
    ]


@pytest.fixture
def dummy_keypoints() -> list:
    return [
        KeyPoint(x=320.0 + i * 5, y=240.0 + i * 3, confidence=0.9,
                 keypoint_id=i, keypoint_name=f"kp_{i}")
        for i in range(17)
    ]


@pytest.fixture
def dummy_pose(dummy_keypoints) -> Pose:
    return Pose(
        bbox=(200.0, 100.0, 440.0, 460.0),
        confidence=0.88, keypoints=dummy_keypoints,
    )
