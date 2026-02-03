"""
Basic tests for PoseEstimator.

These tests verify:
- PoseEstimator initialization (even if model loading fails)
- PoseEstimator process returns expected structure
- Pose data structure integrity
"""

from typing import Dict, Any

import numpy as np

from visionframework import PoseEstimator, Pose, KeyPoint


def _make_dummy_image(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_pose_estimator_creation_returns_instance_even_if_init_fails() -> None:
    """
    PoseEstimator should return an instance even if model loading fails.
    Initialization failures are handled internally via logging.
    """
    config = {
        "model_path": "nonexistent_model.pt",  # Intentionally non-existent path
        "device": "cpu",
        "conf_threshold": 0.5,
    }
    pose_estimator = PoseEstimator(config)
    assert isinstance(pose_estimator, PoseEstimator)


def test_pose_estimator_process_returns_expected_structure() -> None:
    """
    PoseEstimator.process should return a list of Pose objects even if initialization fails.
    """
    config = {
        "model_path": "nonexistent_model.pt",  # Intentionally non-existent path
        "device": "cpu",
        "conf_threshold": 0.5,
    }
    pose_estimator = PoseEstimator(config)
    
    # Initialize (should fail due to non-existent model)
    pose_estimator.initialize()
    
    # Process dummy image
    img = _make_dummy_image()
    poses = pose_estimator.process(img)
    
    # Should return a list (possibly empty)
    assert isinstance(poses, list)
    
    # If any poses are returned, verify they are Pose objects
    for pose in poses:
        assert isinstance(pose, Pose)


def test_pose_data_structure_integrity() -> None:
    """
    Test Pose and KeyPoint data structures for integrity.
    """
    # Create a dummy pose with keypoints
    keypoints = [
        KeyPoint(x=100, y=100, confidence=0.9, keypoint_id=0, keypoint_name="nose"),
        KeyPoint(x=120, y=90, confidence=0.8, keypoint_id=1, keypoint_name="left_eye"),
        KeyPoint(x=80, y=90, confidence=0.8, keypoint_id=2, keypoint_name="right_eye"),
    ]
    
    pose = Pose(
        bbox=(80, 80, 140, 200),
        keypoints=keypoints,
        confidence=0.85,
        pose_id=1
    )
    
    # Verify pose attributes
    assert isinstance(pose, Pose)
    assert isinstance(pose.keypoints, list)
    assert isinstance(pose.confidence, float)
    assert isinstance(pose.bbox, tuple)
    assert isinstance(pose.pose_id, int)
    
    # Verify keypoint attributes
    for keypoint in pose.keypoints:
        assert isinstance(keypoint, KeyPoint)
        assert isinstance(keypoint.x, (int, float))
        assert isinstance(keypoint.y, (int, float))
        assert isinstance(keypoint.confidence, float)
        assert isinstance(keypoint.keypoint_id, int)
        assert isinstance(keypoint.keypoint_name, str)
