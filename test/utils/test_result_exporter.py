"""
ResultExporter 基础测试。

验证：
- ResultExporter 初始化
- 导出功能
- 导出结果格式完整性
"""

import os
import tempfile

from visionframework import ResultExporter, Detection, Track


def _make_dummy_detections() -> list:
    return [
        Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="person"),
        Detection(bbox=(300, 150, 400, 250), confidence=0.85, class_id=1, class_name="car"),
    ]


def _make_dummy_tracks() -> list:
    return [
        Track(track_id=1, bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="person"),
        Track(track_id=2, bbox=(300, 150, 400, 250), confidence=0.85, class_id=1, class_name="car"),
    ]


def test_result_exporter_initialization() -> None:
    """ResultExporter 应能成功初始化。"""
    exporter = ResultExporter()
    assert isinstance(exporter, ResultExporter)


def test_result_exporter_export_detections_to_json() -> None:
    """export_detections_to_json 应导出检测结果为 JSON。"""
    exporter = ResultExporter()
    detections = _make_dummy_detections()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        success = exporter.export_detections_to_json(detections, path)
        assert success
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_result_exporter_export_tracks_to_json() -> None:
    """export_tracks_to_json 应导出跟踪结果为 JSON。"""
    exporter = ResultExporter()
    tracks = _make_dummy_tracks()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        success = exporter.export_tracks_to_json(tracks, path)
        assert success
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_result_exporter_export_detections_to_csv() -> None:
    """export_detections_to_csv 应导出检测结果为 CSV。"""
    exporter = ResultExporter()
    detections = _make_dummy_detections()

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name

    try:
        success = exporter.export_detections_to_csv(detections, path)
        assert success
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_result_exporter_export_tracks_to_csv() -> None:
    """export_tracks_to_csv 应导出跟踪结果为 CSV。"""
    exporter = ResultExporter()
    tracks = _make_dummy_tracks()

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name

    try:
        success = exporter.export_tracks_to_csv(tracks, path)
        assert success
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    finally:
        if os.path.exists(path):
            os.unlink(path)
