"""Tests for the builder module."""

import os
import tempfile

import torch
import pytest

from visionframework.core.builder import build_model, build_model_from_file, ModelWrapper
import visionframework.models  # noqa: F401


class TestBuildModel:
    def test_build_detection_model(self):
        cfg = {
            "backbone": {"type": "CSPDarknet", "depth": 0.33, "width": 0.25},
            "neck": {"type": "PAN", "in_channels": [64, 128, 256], "depth": 0.33},
            "head": {"type": "YOLOHead", "in_channels": [64, 128, 256], "num_classes": 80},
        }
        model = build_model(cfg)
        assert isinstance(model, ModelWrapper)
        assert model.backbone is not None
        assert model.neck is not None
        assert model.head is not None

    def test_build_model_without_neck(self):
        cfg = {
            "backbone": {"type": "CSPDarknet", "depth": 0.33, "width": 0.25},
            "head": {"type": "ReIDHead", "in_channels": [64, 128, 256], "embedding_dim": 256},
        }
        model = build_model(cfg)
        assert model.neck is None

    def test_forward_pass(self):
        cfg = {
            "backbone": {"type": "CSPDarknet", "depth": 0.33, "width": 0.25},
            "neck": {"type": "PAN", "in_channels": [64, 128, 256], "depth": 0.33},
            "head": {"type": "YOLOHead", "in_channels": [64, 128, 256], "num_classes": 80},
        }
        model = build_model(cfg)
        x = torch.randn(1, 3, 640, 640)
        output = model(x)
        assert isinstance(output, list)
        assert len(output) == 3
        for cls_out, reg_out in output:
            assert cls_out.shape[1] == 80

    def test_build_segmentation_model(self):
        cfg = {
            "backbone": {"type": "ResNet", "layers": 18},
            "neck": {"type": "FPN", "in_channels": [128, 256, 512], "out_channels": 128},
            "head": {"type": "SegHead", "in_channels": [128, 128, 128], "num_classes": 21, "hidden_dim": 128},
        }
        model = build_model(cfg)
        x = torch.randn(1, 3, 256, 256)
        output = model(x)
        assert output.shape[1] == 21

    def test_build_reid_model(self):
        cfg = {
            "backbone": {"type": "ResNet", "layers": 18},
            "head": {"type": "ReIDHead", "in_channels": [128, 256, 512], "embedding_dim": 256},
        }
        model = build_model(cfg)
        x = torch.randn(2, 3, 256, 128)
        output = model(x)
        assert output.shape == (2, 256)
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)


class TestBuildYOLO:
    def test_build_yolo11n(self):
        cfg = {
            "backbone": {"type": "YOLOBackbone", "depth": 0.5, "width": 0.25, "max_channels": 1024},
            "neck": {"type": "YOLOPAN", "in_channels": [128, 128, 256], "depth": 0.5, "c3k": False},
            "head": {"type": "YOLOHead", "in_channels": [128, 128, 256], "num_classes": 80, "reg_max": 16},
        }
        model = build_model(cfg)
        x = torch.randn(1, 3, 640, 640)
        output = model(x)
        assert len(output) == 3
        assert output[0][0].shape[1] == 80

    def test_build_yolo11s(self):
        cfg = {
            "backbone": {"type": "YOLOBackbone", "depth": 0.5, "width": 0.5, "max_channels": 1024},
            "neck": {"type": "YOLOPAN", "in_channels": [256, 256, 512], "depth": 0.5, "c3k": False},
            "head": {"type": "YOLOHead", "in_channels": [256, 256, 512], "num_classes": 80, "reg_max": 16},
        }
        model = build_model(cfg)
        x = torch.randn(1, 3, 640, 640)
        output = model(x)
        assert len(output) == 3

    def test_build_yolo26n(self):
        cfg = {
            "backbone": {"type": "YOLOBackbone", "depth": 0.5, "width": 0.25, "max_channels": 1024},
            "neck": {"type": "YOLOPAN", "in_channels": [128, 128, 256], "depth": 0.5, "c3k": True},
            "head": {"type": "YOLOHead", "in_channels": [128, 128, 256], "num_classes": 80, "reg_max": 1},
        }
        model = build_model(cfg)
        x = torch.randn(1, 3, 640, 640)
        output = model(x)
        assert len(output) == 3
        assert output[0][1].shape[1] == 4  # reg_max=1 → 4*1=4

    def test_build_yolo26s(self):
        cfg = {
            "backbone": {"type": "YOLOBackbone", "depth": 0.5, "width": 0.5, "max_channels": 1024},
            "neck": {"type": "YOLOPAN", "in_channels": [256, 256, 512], "depth": 0.5, "c3k": True},
            "head": {"type": "YOLOHead", "in_channels": [256, 256, 512], "num_classes": 80, "reg_max": 1},
        }
        model = build_model(cfg)
        x = torch.randn(1, 3, 640, 640)
        output = model(x)
        assert len(output) == 3


class TestBuildDETR:
    def test_build_detr(self):
        cfg = {
            "backbone": {"type": "ResNet", "layers": 18},
            "neck": {"type": "TransformerEncoderNeck", "in_channels": [128, 256, 512],
                     "d_model": 64, "nhead": 4, "num_layers": 1, "dim_feedforward": 128},
            "head": {"type": "DETRHead", "d_model": 64, "nhead": 4, "num_layers": 1,
                     "num_queries": 10, "num_classes": 20, "dim_feedforward": 128},
        }
        model = build_model(cfg)
        x = torch.randn(1, 3, 256, 256)
        cls_logits, bbox_pred = model(x)
        assert cls_logits.shape == (1, 10, 21)
        assert bbox_pred.shape == (1, 10, 4)


class TestWeightLoading:
    def _make_cfg(self):
        return {
            "backbone": {"type": "YOLOBackbone", "depth": 0.5, "width": 0.25, "max_channels": 1024},
            "neck": {"type": "YOLOPAN", "in_channels": [128, 128, 256], "depth": 0.5, "c3k": False},
            "head": {"type": "YOLOHead", "in_channels": [128, 128, 256], "num_classes": 80, "reg_max": 16},
        }

    def test_load_state_dict(self, tmp_path):
        cfg = self._make_cfg()
        model = build_model(cfg)
        path = str(tmp_path / "weights.pt")
        torch.save(model.state_dict(), path)

        cfg_w = {**cfg, "weights": path, "weights_strict": True}
        model2 = build_model(cfg_w)
        for (n1, p1), (n2, p2) in zip(model.state_dict().items(), model2.state_dict().items()):
            assert torch.equal(p1, p2), f"Mismatch at {n1}"

    def test_load_checkpoint_dict(self, tmp_path):
        cfg = self._make_cfg()
        model = build_model(cfg)
        ckpt = {"model": model.state_dict(), "epoch": 5}
        path = str(tmp_path / "ckpt.pt")
        torch.save(ckpt, path)

        model2 = build_model({**cfg, "weights": path})
        for (n1, p1), (n2, p2) in zip(model.state_dict().items(), model2.state_dict().items()):
            assert torch.equal(p1, p2)

    def test_missing_weights_file(self):
        cfg = {**self._make_cfg(), "weights": "/nonexistent/path.pt"}
        model = build_model(cfg)
        assert isinstance(model, ModelWrapper)

    def test_weights_override(self, tmp_path):
        cfg = self._make_cfg()
        model = build_model(cfg)
        path = str(tmp_path / "w.pt")
        torch.save(model.state_dict(), path)

        model2 = build_model(cfg, weights=path)
        for (n1, p1), (n2, p2) in zip(model.state_dict().items(), model2.state_dict().items()):
            assert torch.equal(p1, p2)
