"""Tests for pipeline modules."""

import numpy as np
import torch
import pytest
import tempfile

from visionframework.core.builder import build_model
from visionframework.algorithms.detection.detector import Detector
from visionframework.algorithms.tracking.byte_tracker import ByteTracker
from visionframework.algorithms.tracking.iou_tracker import IOUTracker
from visionframework.pipelines.detection_pipeline import DetectionPipeline
from visionframework.pipelines.tracking_pipeline import TrackingPipeline
from visionframework import TaskRunner
import visionframework.models  # noqa: F401


def _make_detector():
    cfg = {
        "backbone": {"type": "CSPDarknet", "depth": 0.33, "width": 0.25},
        "neck": {"type": "PAN", "in_channels": [64, 128, 256], "depth": 0.33},
        "head": {"type": "YOLOHead", "in_channels": [64, 128, 256], "num_classes": 80},
    }
    model = build_model(cfg)
    return Detector(model=model, device="cpu", conf=0.001, nms_iou=0.45)


class TestDetectionPipeline:
    def test_process_returns_dict(self):
        det = _make_detector()
        pipe = DetectionPipeline(detector=det)
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = pipe.process(frame)
        assert "detections" in result
        assert isinstance(result["detections"], list)


class TestTrackingPipeline:
    def test_process_returns_detections_and_tracks(self):
        det = _make_detector()
        tracker = ByteTracker(track_thresh=0.001)
        pipe = TrackingPipeline(detector=det, tracker=tracker)
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = pipe.process(frame)
        assert "detections" in result
        assert "tracks" in result

    def test_reset_clears_tracker(self):
        det = _make_detector()
        tracker = ByteTracker()
        pipe = TrackingPipeline(detector=det, tracker=tracker)
        pipe.process(np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8))
        pipe.reset()
        assert tracker.frame_id == 0


class TestTaskRunnerWeights:
    """TaskRunner 从 runtime YAML 透传 weights 的集成测试。"""

    def _make_runtime_yaml(self, tmp_path, weights_path):
        import yaml
        model_cfg = {
            "task": "detection",
            "backbone": {"type": "CSPDarknet", "depth": 0.33, "width": 0.25},
            "neck": {"type": "PAN", "in_channels": [64, 128, 256], "depth": 0.33},
            "head": {"type": "YOLOHead", "in_channels": [64, 128, 256], "num_classes": 80},
        }
        model_yaml = str(tmp_path / "model.yaml")
        with open(model_yaml, "w") as f:
            yaml.dump(model_cfg, f)

        runtime_cfg = {
            "pipeline": "detection",
            "models": {"detector": model_yaml},
            "weights": weights_path,
            "device": "cpu",
        }
        runtime_yaml = str(tmp_path / "runtime.yaml")
        with open(runtime_yaml, "w") as f:
            yaml.dump(runtime_cfg, f)
        return runtime_yaml

    def test_weights_loaded_via_taskrunner(self, tmp_path):
        """TaskRunner 应正确从 runtime YAML 的 weights 字段加载权重。"""
        # 先构建一个模型并保存权重
        cfg = {
            "backbone": {"type": "CSPDarknet", "depth": 0.33, "width": 0.25},
            "neck": {"type": "PAN", "in_channels": [64, 128, 256], "depth": 0.33},
            "head": {"type": "YOLOHead", "in_channels": [64, 128, 256], "num_classes": 80},
        }
        model = build_model(cfg)
        weights_path = str(tmp_path / "weights.pth")
        torch.save(model.state_dict(), weights_path)

        runtime_yaml = self._make_runtime_yaml(tmp_path, weights_path)
        task = TaskRunner(runtime_yaml)

        # 验证模型权重与保存的一致
        loaded_sd = task.pipeline.detector.model.state_dict()
        for k, v in model.state_dict().items():
            assert torch.equal(v, loaded_sd[k]), f"权重不一致: {k}"

    def test_dict_weights_resolves_detector(self, tmp_path):
        """weights 为 dict 时，detector 角色正确解析。"""
        import yaml
        cfg = {
            "backbone": {"type": "CSPDarknet", "depth": 0.33, "width": 0.25},
            "neck": {"type": "PAN", "in_channels": [64, 128, 256], "depth": 0.33},
            "head": {"type": "YOLOHead", "in_channels": [64, 128, 256], "num_classes": 80},
        }
        model = build_model(cfg)
        weights_path = str(tmp_path / "det.pth")
        torch.save(model.state_dict(), weights_path)

        model_yaml = str(tmp_path / "model.yaml")
        with open(model_yaml, "w") as f:
            yaml.dump({"task": "detection", **cfg}, f)

        runtime_cfg = {
            "pipeline": "detection",
            "models": {"detector": model_yaml},
            "weights": {"detector": weights_path},
            "device": "cpu",
        }
        runtime_yaml = str(tmp_path / "runtime.yaml")
        with open(runtime_yaml, "w") as f:
            yaml.dump(runtime_cfg, f)

        task = TaskRunner(runtime_yaml)
        loaded_sd = task.pipeline.detector.model.state_dict()
        for k, v in model.state_dict().items():
            assert torch.equal(v, loaded_sd[k])
