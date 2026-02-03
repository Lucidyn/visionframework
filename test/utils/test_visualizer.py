"""
Basic tests for Visualizer.

These tests verify:
- Visualizer initialization
- Various drawing methods return expected structure
- Visualization output integrity
"""

import numpy as np

from visionframework.utils.visualization.unified_visualizer import Visualizer
from visionframework.data.detection import Detection
from visionframework.data.track import Track
from visionframework.data.pose import Pose, KeyPoint


def _make_dummy_image(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_dummy_detections() -> list:
    """Create dummy detections for testing."""
    return [
        Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="person"
        ),
        Detection(
            bbox=(300, 150, 400, 250),
            confidence=0.85,
            class_id=1,
            class_name="car"
        )
    ]


def _make_dummy_tracks() -> list:
    """Create dummy tracks for testing."""
    return [
        Track(
            track_id=1,
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="person"
        ),
        Track(
            track_id=2,
            bbox=(300, 150, 400, 250),
            confidence=0.85,
            class_id=1,
            class_name="car"
        )
    ]


def _make_dummy_poses() -> list:
    """Create dummy poses for testing."""
    return [
        Pose(
            bbox=(100, 100, 200, 300),
            keypoints=[
                KeyPoint(keypoint_id=0, keypoint_name="nose", x=150, y=120, confidence=0.9),
                KeyPoint(keypoint_id=1, keypoint_name="left_eye", x=130, y=110, confidence=0.8),
                KeyPoint(keypoint_id=2, keypoint_name="right_eye", x=170, y=110, confidence=0.8),
                KeyPoint(keypoint_id=3, keypoint_name="neck", x=150, y=180, confidence=0.85),
                KeyPoint(keypoint_id=4, keypoint_name="left_shoulder", x=120, y=220, confidence=0.8),
                KeyPoint(keypoint_id=5, keypoint_name="right_shoulder", x=180, y=220, confidence=0.8)
            ],
            confidence=0.85,
            pose_id=1
        )
    ]


def test_visualizer_initialization() -> None:
    """
    Visualizer should initialize successfully with default or custom parameters.
    """
    # Test default initialization
    visualizer = Visualizer()
    assert isinstance(visualizer, Visualizer)
    
    # Test custom initialization
    custom_config = {
        "line_thickness": 2,
        "font_scale": 0.5,
        "font_thickness": 1,
        "colors": {0: (0, 255, 0), 1: (0, 0, 255)}
    }
    custom_visualizer = Visualizer(config=custom_config)
    assert isinstance(custom_visualizer, Visualizer)


def test_visualizer_draw_detections() -> None:
    """
    Visualizer.draw_detections should return an image array.
    """
    visualizer = Visualizer()
    img = _make_dummy_image()
    detections = _make_dummy_detections()
    
    # Draw detections
    vis_image = visualizer.draw_detections(img.copy(), detections)
    
    # Should return a numpy array with the same shape as input
    assert isinstance(vis_image, np.ndarray)
    assert vis_image.shape == img.shape


def test_visualizer_draw_tracks() -> None:
    """
    Visualizer.draw_tracks should return an image array.
    """
    visualizer = Visualizer()
    img = _make_dummy_image()
    tracks = _make_dummy_tracks()
    
    # Draw tracks
    vis_image = visualizer.draw_tracks(img.copy(), tracks)
    
    # Should return a numpy array with the same shape as input
    assert isinstance(vis_image, np.ndarray)
    assert vis_image.shape == img.shape


def test_visualizer_draw_poses() -> None:
    """
    Visualizer.draw_poses should return an image array.
    """
    visualizer = Visualizer()
    img = _make_dummy_image()
    poses = _make_dummy_poses()
    
    # Draw poses
    vis_image = visualizer.draw_poses(img.copy(), poses)
    
    # Should return a numpy array with the same shape as input
    assert isinstance(vis_image, np.ndarray)
    assert vis_image.shape == img.shape


def test_visualizer_draw_results() -> None:
    """
    Visualizer.draw_results should return an image array with all results drawn.
    """
    visualizer = Visualizer()
    img = _make_dummy_image()
    detections = _make_dummy_detections()
    tracks = _make_dummy_tracks()
    poses = _make_dummy_poses()
    
    # Draw all results
    vis_image = visualizer.draw_results(
        img.copy(),
        detections=detections,
        tracks=tracks,
        poses=poses
    )
    
    # Should return a numpy array with the same shape as input
    assert isinstance(vis_image, np.ndarray)
    assert vis_image.shape == img.shape


def test_visualizer_with_empty_inputs() -> None:
    """
    Visualizer should handle empty inputs gracefully.
    """
    visualizer = Visualizer()
    img = _make_dummy_image()
    
    # Draw with empty detections
    vis_image_empty_detections = visualizer.draw_detections(img.copy(), [])
    assert isinstance(vis_image_empty_detections, np.ndarray)
    assert vis_image_empty_detections.shape == img.shape
    
    # Draw with empty tracks
    vis_image_empty_tracks = visualizer.draw_tracks(img.copy(), [])
    assert isinstance(vis_image_empty_tracks, np.ndarray)
    assert vis_image_empty_tracks.shape == img.shape
    
    # Draw with empty poses
    vis_image_empty_poses = visualizer.draw_poses(img.copy(), [])
    assert isinstance(vis_image_empty_poses, np.ndarray)
    assert vis_image_empty_poses.shape == img.shape
    
    # Draw with all empty inputs
    vis_image_empty_all = visualizer.draw_results(
        img.copy(),
        detections=[],
        tracks=[],
        poses=[]
    )
    assert isinstance(vis_image_empty_all, np.ndarray)
    assert vis_image_empty_all.shape == img.shape
