"""
Basic tests for VideoPipeline.

These tests verify:
- VideoPipeline initialization (even if model loading fails)
- VideoPipeline process returns expected structure
- VideoPipeline functionality with different configurations
"""

import numpy as np

from visionframework import VideoPipeline


def _make_dummy_image(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_video_pipeline_creation_returns_instance_even_if_init_fails() -> None:
    """
    VideoPipeline should return an instance even if model loading fails.
    Initialization failures are handled internally via logging.
    """
    config = {
        "detector_config": {
            "model_path": "nonexistent_model.pt",  # Intentionally non-existent path
            "device": "cpu",
            "conf_threshold": 0.5,
        },
        "enable_tracking": True,
        "tracker_config": {
            "tracker_type": "bytetrack",
        },
    }
    video_pipeline = VideoPipeline(config)
    assert isinstance(video_pipeline, VideoPipeline)


def test_video_pipeline_process_returns_expected_structure() -> None:
    """
    VideoPipeline.process should return expected structure even if initialization fails.
    """
    config = {
        "detector_config": {
            "model_path": "nonexistent_model.pt",  # Intentionally non-existent path
            "device": "cpu",
            "conf_threshold": 0.5,
        },
        "enable_tracking": True,
        "tracker_config": {
            "tracker_type": "bytetrack",
        },
    }
    video_pipeline = VideoPipeline(config)
    
    # Initialize (should fail due to non-existent model)
    video_pipeline.initialize()
    
    # Process dummy frame
    frame = _make_dummy_image()
    result = video_pipeline.process(frame)
    
    # Should return a dict with expected keys
    assert isinstance(result, dict)
    assert "detections" in result
    assert "tracks" in result
    assert "poses" in result
    assert isinstance(result["detections"], list)
    assert isinstance(result["tracks"], list)
    assert isinstance(result["poses"], list)


def test_video_pipeline_with_pose_estimation() -> None:
    """
    Test VideoPipeline with pose estimation enabled.
    """
    config = {
        "detector_config": {
            "model_path": "nonexistent_model.pt",  # Intentionally non-existent path
            "device": "cpu",
            "conf_threshold": 0.5,
        },
        "enable_tracking": False,
        "enable_pose_estimation": True,
        "pose_estimator_config": {
            "model_path": "nonexistent_model.pt",  # Intentionally non-existent path
            "device": "cpu",
            "conf_threshold": 0.5,
        },
    }
    video_pipeline = VideoPipeline(config)
    
    # Initialize (should fail due to non-existent models)
    video_pipeline.initialize()
    
    # Process dummy frame
    frame = _make_dummy_image()
    result = video_pipeline.process(frame)
    
    # Should return a dict with expected keys
    assert isinstance(result, dict)
    assert "detections" in result
    assert "tracks" in result
    assert "poses" in result
    assert isinstance(result["detections"], list)
    assert isinstance(result["tracks"], list)
    assert isinstance(result["poses"], list)


def test_video_pipeline_with_segmentation() -> None:
    """
    Test VideoPipeline with segmentation enabled.
    """
    config = {
        "detector_config": {
            "model_path": "nonexistent_model.pt",  # Intentionally non-existent path
            "device": "cpu",
            "conf_threshold": 0.5,
        },
        "enable_tracking": False,
        "enable_segmentation": True,
        "segmenter_config": {
            "model_path": "nonexistent_model.pt",  # Intentionally non-existent path
            "device": "cpu",
            "conf_threshold": 0.5,
        },
    }
    video_pipeline = VideoPipeline(config)
    
    # Initialize (should fail due to non-existent models)
    video_pipeline.initialize()
    
    # Process dummy frame
    frame = _make_dummy_image()
    result = video_pipeline.process(frame)
    
    # Should return a dict with expected keys
    assert isinstance(result, dict)
    assert "detections" in result
    assert "tracks" in result
    assert "poses" in result
    assert isinstance(result["detections"], list)
    assert isinstance(result["tracks"], list)
    assert isinstance(result["poses"], list)
