"""Tests for visualization module."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from visionframework.data.detection import Detection
from visionframework.utils.visualization import Visualizer

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TEST_BUS = _REPO_ROOT / "test_bus.jpg"


class TestVisualizer:
    def test_init(self):
        viz = Visualizer()
        assert viz is not None

    def test_draw_detections(self, dummy_image, dummy_detections):
        viz = Visualizer()
        result = viz.draw_detections(dummy_image, dummy_detections)
        assert result.shape == dummy_image.shape

    def test_draw_tracks(self, dummy_image, dummy_tracks):
        viz = Visualizer()
        result = viz.draw_tracks(dummy_image, dummy_tracks)
        assert result.shape == dummy_image.shape

    def test_draw_results_combined(self, dummy_image, dummy_detections, dummy_tracks):
        viz = Visualizer()
        result = viz.draw_results(dummy_image, detections=dummy_detections, tracks=dummy_tracks)
        assert result.shape == dummy_image.shape

    def test_draw_empty(self, dummy_image):
        viz = Visualizer()
        result = viz.draw_results(dummy_image)
        assert result.shape == dummy_image.shape

    def test_draw_from_result_dict(self, dummy_image, dummy_detections):
        viz = Visualizer()
        result_dict = {"detections": dummy_detections}
        result = viz.draw(dummy_image, result_dict)
        assert result.shape == dummy_image.shape

    def test_draw_detections_writes_image(self, dummy_image, dummy_detections, tmp_path):
        """OpenCV 写出 JPEG，验收可视化管线可生成非空图像文件。"""
        viz = Visualizer()
        drawn = viz.draw_detections(dummy_image, dummy_detections)
        out = tmp_path / "viz.jpg"
        assert cv2.imwrite(str(out), drawn)
        assert out.stat().st_size > 500
        loaded = cv2.imread(str(out))
        assert loaded is not None
        assert loaded.shape == dummy_image.shape

    @pytest.mark.skipif(not _TEST_BUS.is_file(), reason="test_bus.jpg missing (Ultralytics bus sample at repo root)")
    def test_draw_on_real_test_bus_sample(self, tmp_path):
        """在真实样图上绘制，验收分辨率与 OpenCV 管线。"""
        img = cv2.imread(str(_TEST_BUS))
        assert img is not None
        h, w = img.shape[:2]
        dets = [
            Detection(bbox=(48, int(h * 0.38), 245, int(h * 0.82)), confidence=0.9, class_id=0, class_name="person"),
        ]
        viz = Visualizer()
        drawn = viz.draw_detections(img, dets)
        out = tmp_path / "bus_viz.jpg"
        assert cv2.imwrite(str(out), drawn)
        assert out.stat().st_size > 1000
