"""
Basic tests for ResultExporter.

These tests verify:
- ResultExporter initialization
- ResultExporter export functionality
- Export result format integrity
"""

import os
import tempfile

from visionframework.utils.data.export import ResultExporter
from visionframework.data.detection import Detection
from visionframework.data.track import Track


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


def test_result_exporter_initialization() -> None:
    """
    ResultExporter should initialize successfully.
    """
    # Test default initialization
    exporter = ResultExporter()
    assert isinstance(exporter, ResultExporter)


def test_result_exporter_export_detections_to_json() -> None:
    """
    ResultExporter.export_detections_to_json should export detections to JSON format.
    """
    exporter = ResultExporter()
    
    # Create dummy detections
    detections = _make_dummy_detections()
    
    # Export to temporary file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    try:
        # Export to JSON
        success = exporter.export_detections_to_json(detections, temp_file_path)
        assert success
        
        # Verify file exists
        assert os.path.exists(temp_file_path)
        assert os.path.getsize(temp_file_path) > 0
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def test_result_exporter_export_tracks_to_json() -> None:
    """
    ResultExporter.export_tracks_to_json should export tracks to JSON format.
    """
    exporter = ResultExporter()
    
    # Create dummy tracks
    tracks = _make_dummy_tracks()
    
    # Export to temporary file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    try:
        # Export to JSON
        success = exporter.export_tracks_to_json(tracks, temp_file_path)
        assert success
        
        # Verify file exists
        assert os.path.exists(temp_file_path)
        assert os.path.getsize(temp_file_path) > 0
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def test_result_exporter_export_detections_to_csv() -> None:
    """
    ResultExporter.export_detections_to_csv should export detections to CSV format.
    """
    exporter = ResultExporter()
    
    # Create dummy detections
    detections = _make_dummy_detections()
    
    # Export to temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    try:
        # Export to CSV
        success = exporter.export_detections_to_csv(detections, temp_file_path)
        assert success
        
        # Verify file exists
        assert os.path.exists(temp_file_path)
        assert os.path.getsize(temp_file_path) > 0
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def test_result_exporter_export_tracks_to_csv() -> None:
    """
    ResultExporter.export_tracks_to_csv should export tracks to CSV format.
    """
    exporter = ResultExporter()
    
    # Create dummy tracks
    tracks = _make_dummy_tracks()
    
    # Export to temporary file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    try:
        # Export to CSV
        success = exporter.export_tracks_to_csv(tracks, temp_file_path)
        assert success
        
        # Verify file exists
        assert os.path.exists(temp_file_path)
        assert os.path.getsize(temp_file_path) > 0
    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
