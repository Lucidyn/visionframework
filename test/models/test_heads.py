"""Tests for head modules."""

import torch

from visionframework.models.heads.yolo_head import YOLOHead
from visionframework.models.heads.seg_head import SegHead
from visionframework.models.heads.reid_head import ReIDHead
from visionframework.models.heads.detr_head import DETRHead
from visionframework.models.heads.rfdetr_head import RFDETRHead


class TestYOLOHead:
    def test_output_per_level(self):
        head = YOLOHead(in_channels=[64, 128, 256], num_classes=80, reg_max=16)
        feats = [torch.randn(1, 64, 80, 80),
                 torch.randn(1, 128, 40, 40),
                 torch.randn(1, 256, 20, 20)]
        out = head(feats)
        assert len(out) == 3
        for cls_out, reg_out in out:
            assert cls_out.shape[1] == 80
            assert reg_out.shape[1] == 4 * 16


class TestSegHead:
    def test_output_classes(self):
        head = SegHead(in_channels=[128, 128, 128], num_classes=21, hidden_dim=64)
        feats = [torch.randn(1, 128, 80, 80),
                 torch.randn(1, 128, 40, 40),
                 torch.randn(1, 128, 20, 20)]
        out = head(feats)
        assert out.shape[1] == 21
        assert out.shape[2:] == (80, 80)


class TestReIDHead:
    def test_embedding_dim(self):
        head = ReIDHead(in_channels=[64, 128, 256], embedding_dim=512)
        feats = [torch.randn(2, 64, 32, 32),
                 torch.randn(2, 128, 16, 16),
                 torch.randn(2, 256, 8, 8)]
        out = head(feats)
        assert out.shape == (2, 512)

    def test_l2_normalized(self):
        head = ReIDHead(in_channels=256, embedding_dim=128)
        feat = torch.randn(4, 256, 8, 8)
        out = head(feat)
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)


class TestDETRHead:
    def test_output_shapes(self):
        head = DETRHead(d_model=64, nhead=4, num_layers=2,
                        num_queries=50, num_classes=80, dim_feedforward=128)
        memory = torch.randn(1, 64, 64)  # (B, HW, d_model)
        pos = torch.randn(1, 64, 64)     # (B, HW, d_model)
        cls_logits, bbox_pred = head((memory, pos, (8, 8)))
        assert cls_logits.shape == (1, 50, 81)  # nc + 1
        assert bbox_pred.shape == (1, 50, 4)

    def test_bbox_normalized(self):
        head = DETRHead(d_model=64, nhead=4, num_layers=1,
                        num_queries=10, num_classes=20, dim_feedforward=64)
        memory = torch.randn(1, 16, 64)
        pos = torch.randn(1, 16, 64)
        _, bbox_pred = head((memory, pos, (4, 4)))
        assert (bbox_pred >= 0).all() and (bbox_pred <= 1).all()


class TestRFDETRHead:
    def test_output_shapes(self):
        head = RFDETRHead(d_model=64, nhead=4, num_layers=2,
                          num_queries=50, num_classes=80, n_points=4,
                          dim_feedforward=128)
        memory = torch.randn(1, 64, 64)
        cls_logits, bbox_pred = head((memory, (8, 8)))
        assert cls_logits.shape == (1, 50, 81)
        assert bbox_pred.shape == (1, 50, 4)
