"""
Basic tests for SAMSegmenter.

These tests verify:
- SAMSegmenter initialization (even if model loading fails)
- SAMSegmenter process returns expected structure
"""

import numpy as np

from visionframework import SAMSegmenter


def _make_dummy_image(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_sam_segmenter_creation_returns_instance_even_if_init_fails() -> None:
    """
    SAMSegmenter should return an instance even if model loading fails.
    Initialization failures are handled internally via logging.
    """
    config = {
        "model_path": "nonexistent_model.pt",  # Intentionally non-existent path
        "device": "cpu",
        "conf_threshold": 0.5,
    }
    segmenter = SAMSegmenter(config)
    assert isinstance(segmenter, SAMSegmenter)


def test_sam_segmenter_process_returns_expected_structure() -> None:
    """
    SAMSegmenter.process should return expected structure even if initialization fails.
    """
    config = {
        "model_path": "nonexistent_model.pt",  # Intentionally non-existent path
        "device": "cpu",
        "conf_threshold": 0.5,
    }
    segmenter = SAMSegmenter(config)
    
    # Initialize (should fail due to non-existent model)
    segmenter.initialize()
    
    # Process dummy image with dummy points
    img = _make_dummy_image()
    points = [(100, 100), (200, 200)]  # Dummy points
    point_labels = [1, 1]  # Dummy point labels
    
    # Test with points
    segments_with_points = segmenter.process(img, points=points, point_labels=point_labels)
    assert isinstance(segments_with_points, list)
    
    # Test without points (auto-segmentation)
    segments_auto = segmenter.process(img)
    assert isinstance(segments_auto, list)


def test_sam_segmenter_with_bbox_prompt() -> None:
    """
    Test SAMSegmenter with bounding box prompt.
    """
    config = {
        "model_path": "nonexistent_model.pt",  # Intentionally non-existent path
        "device": "cpu",
        "conf_threshold": 0.5,
    }
    segmenter = SAMSegmenter(config)
    
    # Initialize (should fail due to non-existent model)
    segmenter.initialize()
    
    # Process dummy image with dummy bbox
    img = _make_dummy_image()
    bbox = (50, 50, 150, 150)  # Dummy bbox
    
    segments = segmenter.process(img, bbox=bbox)
    assert isinstance(segments, list)
