"""Tests for backbone networks."""

import torch
import pytest

from visionframework.models.backbones.cspdarknet import CSPDarknet
from visionframework.models.backbones.resnet import ResNet
from visionframework.models.backbones.yolo import YOLOBackbone


class TestCSPDarknet:
    def test_output_shapes(self):
        net = CSPDarknet(depth=0.33, width=0.25)
        x = torch.randn(1, 3, 640, 640)
        feats = net(x)
        assert len(feats) == 3
        assert feats[0].shape[2:] == (80, 80)
        assert feats[1].shape[2:] == (40, 40)
        assert feats[2].shape[2:] == (20, 20)

    def test_output_channels(self):
        net = CSPDarknet(depth=0.33, width=0.25)
        x = torch.randn(1, 3, 640, 640)
        feats = net(x)
        for feat, ch in zip(feats, net.out_channels):
            assert feat.shape[1] == ch

    def test_different_scales(self):
        for w in [0.25, 0.5, 1.0]:
            net = CSPDarknet(depth=0.33, width=w)
            x = torch.randn(1, 3, 320, 320)
            feats = net(x)
            assert len(feats) == 3


class TestResNet:
    @pytest.mark.parametrize("layers", [18, 34, 50])
    def test_output_shapes(self, layers):
        net = ResNet(layers=layers)
        x = torch.randn(1, 3, 256, 256)
        feats = net(x)
        assert len(feats) == 3
        assert feats[0].shape[2:] == (32, 32)
        assert feats[1].shape[2:] == (16, 16)
        assert feats[2].shape[2:] == (8, 8)

    def test_output_channels(self):
        net = ResNet(layers=50)
        x = torch.randn(1, 3, 256, 256)
        feats = net(x)
        assert feats[0].shape[1] == 512
        assert feats[1].shape[1] == 1024
        assert feats[2].shape[1] == 2048

    def test_invalid_variant(self):
        with pytest.raises(ValueError):
            ResNet(layers=99)


class TestYOLOBackbone:
    def test_nano_shapes(self):
        net = YOLOBackbone(depth=0.5, width=0.25, max_channels=1024)
        x = torch.randn(1, 3, 640, 640)
        feats = net(x)
        assert len(feats) == 3
        assert feats[0].shape == (1, 128, 80, 80)
        assert feats[1].shape == (1, 128, 40, 40)
        assert feats[2].shape == (1, 256, 20, 20)

    def test_small_shapes(self):
        net = YOLOBackbone(depth=0.5, width=0.5, max_channels=1024)
        x = torch.randn(1, 3, 640, 640)
        feats = net(x)
        assert len(feats) == 3
        assert feats[0].shape[1] == 256
        assert feats[1].shape[1] == 256
        assert feats[2].shape[1] == 512

    def test_medium_shapes(self):
        net = YOLOBackbone(depth=0.5, width=1.0, max_channels=512)
        x = torch.randn(1, 3, 320, 320)
        feats = net(x)
        assert len(feats) == 3
        assert feats[0].shape[1] == 512
        assert feats[1].shape[1] == 512
        assert feats[2].shape[1] == 512

    def test_out_channels_match(self):
        net = YOLOBackbone(depth=0.5, width=0.25, max_channels=1024)
        x = torch.randn(1, 3, 640, 640)
        feats = net(x)
        for feat, ch in zip(feats, net.out_channels):
            assert feat.shape[1] == ch

    def test_large_shapes(self):
        net = YOLOBackbone(depth=1.0, width=1.0, max_channels=512)
        x = torch.randn(1, 3, 320, 320)
        feats = net(x)
        assert len(feats) == 3
        for feat, ch in zip(feats, net.out_channels):
            assert feat.shape[1] == ch
