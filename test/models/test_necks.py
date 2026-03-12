"""Tests for neck modules."""

import torch

from visionframework.models.necks.pan import PAN
from visionframework.models.necks.fpn import FPN
from visionframework.models.necks.yolo_pan import YOLOPAN
from visionframework.models.necks.transformer_encoder import TransformerEncoderNeck
from visionframework.models.necks.deformable_encoder import DeformableEncoderNeck


class TestPAN:
    def test_output_count(self):
        neck = PAN(in_channels=[64, 128, 256])
        feats = [torch.randn(1, 64, 80, 80),
                 torch.randn(1, 128, 40, 40),
                 torch.randn(1, 256, 20, 20)]
        out = neck(feats)
        assert len(out) == 3

    def test_preserves_channels(self):
        neck = PAN(in_channels=[64, 128, 256])
        feats = [torch.randn(1, 64, 80, 80),
                 torch.randn(1, 128, 40, 40),
                 torch.randn(1, 256, 20, 20)]
        out = neck(feats)
        assert out[0].shape[1] == 64
        assert out[1].shape[1] == 128
        assert out[2].shape[1] == 256


class TestFPN:
    def test_output_count(self):
        neck = FPN(in_channels=[256, 512, 1024], out_channels=256)
        feats = [torch.randn(1, 256, 80, 80),
                 torch.randn(1, 512, 40, 40),
                 torch.randn(1, 1024, 20, 20)]
        out = neck(feats)
        assert len(out) == 3

    def test_uniform_channels(self):
        neck = FPN(in_channels=[256, 512, 1024], out_channels=128)
        feats = [torch.randn(1, 256, 80, 80),
                 torch.randn(1, 512, 40, 40),
                 torch.randn(1, 1024, 20, 20)]
        out = neck(feats)
        for f in out:
            assert f.shape[1] == 128


class TestYOLOPAN:
    def test_output_count_c3k_false(self):
        neck = YOLOPAN(in_channels=[128, 128, 256], depth=0.5, c3k=False)
        feats = [torch.randn(1, 128, 80, 80),
                 torch.randn(1, 128, 40, 40),
                 torch.randn(1, 256, 20, 20)]
        out = neck(feats)
        assert len(out) == 3

    def test_preserves_channels_c3k_false(self):
        neck = YOLOPAN(in_channels=[128, 128, 256], depth=0.5, c3k=False)
        feats = [torch.randn(1, 128, 80, 80),
                 torch.randn(1, 128, 40, 40),
                 torch.randn(1, 256, 20, 20)]
        out = neck(feats)
        assert out[0].shape[1] == 128
        assert out[1].shape[1] == 128
        assert out[2].shape[1] == 256

    def test_spatial_sizes(self):
        neck = YOLOPAN(in_channels=[128, 128, 256], depth=0.5, c3k=False)
        feats = [torch.randn(1, 128, 80, 80),
                 torch.randn(1, 128, 40, 40),
                 torch.randn(1, 256, 20, 20)]
        out = neck(feats)
        assert out[0].shape[2:] == (80, 80)
        assert out[1].shape[2:] == (40, 40)
        assert out[2].shape[2:] == (20, 20)

    def test_output_count_c3k_true(self):
        neck = YOLOPAN(in_channels=[128, 128, 256], depth=0.5, c3k=True)
        feats = [torch.randn(1, 128, 80, 80),
                 torch.randn(1, 128, 40, 40),
                 torch.randn(1, 256, 20, 20)]
        out = neck(feats)
        assert len(out) == 3

    def test_preserves_channels_c3k_true(self):
        neck = YOLOPAN(in_channels=[256, 256, 512], depth=0.5, c3k=True)
        feats = [torch.randn(1, 256, 80, 80),
                 torch.randn(1, 256, 40, 40),
                 torch.randn(1, 512, 20, 20)]
        out = neck(feats)
        assert out[0].shape[1] == 256
        assert out[1].shape[1] == 256
        assert out[2].shape[1] == 512


class TestTransformerEncoderNeck:
    def test_output_shape(self):
        neck = TransformerEncoderNeck(
            in_channels=[512, 1024, 2048], d_model=256,
            nhead=8, num_layers=2, dim_feedforward=512
        )
        feats = [torch.randn(1, 512, 32, 32),
                 torch.randn(1, 1024, 16, 16),
                 torch.randn(1, 2048, 8, 8)]
        memory, pos, (H, W) = neck(feats)
        assert memory.shape == (1, H * W, 256)
        assert pos.shape == memory.shape
        assert H == 8 and W == 8


class TestDeformableEncoderNeck:
    def test_output_shape(self):
        neck = DeformableEncoderNeck(
            in_channels=[256], d_model=256,
            nhead=8, num_layers=2, n_points=4
        )
        feats = [torch.randn(1, 256, 8, 8)]
        memory, (H, W) = neck(feats)
        assert memory.shape == (1, 64, 256)
        assert H == 8 and W == 8
