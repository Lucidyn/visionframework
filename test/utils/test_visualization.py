"""Tests for visualization module."""

import numpy as np
from visionframework.data.detection import Detection
from visionframework.data.track import Track
from visionframework.data.pose import Pose, KeyPoint
from visionframework.utils.visualization import Visualizer


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
