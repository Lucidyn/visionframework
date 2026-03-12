"""Tests for atomic network layers."""

import torch

from visionframework.layers import (
    ConvBNAct, DWConvBNAct, DepthwiseSepConv, Focus,
    Bottleneck, CSPBlock, C2f, C3k, C3k2, Attention, PSABlock, C2PSA,
    SPPF, SPP,
    SEBlock, CBAM, TransformerBlock,
    PositionalEncoding2D, DeformableAttention,
    MLP,
)


class TestConvLayers:
    def test_conv_bn_act(self):
        m = ConvBNAct(3, 16, 3, 1)
        x = torch.randn(1, 3, 64, 64)
        assert m(x).shape == (1, 16, 64, 64)

    def test_dw_conv_bn_act(self):
        m = DWConvBNAct(16, 16, 3, 1)
        x = torch.randn(1, 16, 32, 32)
        assert m(x).shape == (1, 16, 32, 32)

    def test_depthwise_sep_conv(self):
        m = DepthwiseSepConv(16, 32, 3, 1)
        x = torch.randn(1, 16, 32, 32)
        assert m(x).shape == (1, 32, 32, 32)

    def test_focus(self):
        m = Focus(3, 32)
        x = torch.randn(1, 3, 64, 64)
        assert m(x).shape == (1, 32, 32, 32)


class TestCSPLayers:
    def test_bottleneck(self):
        m = Bottleneck(64, 64, shortcut=True)
        x = torch.randn(1, 64, 32, 32)
        assert m(x).shape == x.shape

    def test_bottleneck_custom_kernel(self):
        m = Bottleneck(64, 64, shortcut=True, k=(5, 5))
        x = torch.randn(1, 64, 32, 32)
        assert m(x).shape == x.shape

    def test_csp_block(self):
        m = CSPBlock(64, 64, n=2)
        x = torch.randn(1, 64, 32, 32)
        assert m(x).shape == (1, 64, 32, 32)

    def test_c2f(self):
        m = C2f(64, 64, n=2)
        x = torch.randn(1, 64, 32, 32)
        assert m(x).shape == (1, 64, 32, 32)


class TestC3k2Layers:
    def test_c3k_basic(self):
        m = C3k(64, 64, n=1, k=3)
        x = torch.randn(1, 64, 32, 32)
        assert m(x).shape == (1, 64, 32, 32)

    def test_c3k2_without_c3k(self):
        m = C3k2(64, 128, n=1, c3k=False, e=0.5)
        x = torch.randn(1, 64, 32, 32)
        assert m(x).shape == (1, 128, 32, 32)

    def test_c3k2_with_c3k(self):
        m = C3k2(128, 128, n=1, c3k=True)
        x = torch.randn(1, 128, 32, 32)
        assert m(x).shape == (1, 128, 32, 32)

    def test_c3k2_multiple_repeats(self):
        m = C3k2(64, 64, n=3, c3k=True)
        x = torch.randn(1, 64, 16, 16)
        assert m(x).shape == (1, 64, 16, 16)

    def test_c3k2_low_expansion(self):
        m = C3k2(32, 64, n=1, c3k=False, e=0.25)
        x = torch.randn(1, 32, 32, 32)
        assert m(x).shape == (1, 64, 32, 32)


class TestAttentionModule:
    def test_attention_forward(self):
        m = Attention(128, num_heads=4, attn_ratio=0.5)
        x = torch.randn(1, 128, 16, 16)
        assert m(x).shape == (1, 128, 16, 16)

    def test_psablock_forward(self):
        m = PSABlock(128, attn_ratio=0.5, num_heads=4)
        x = torch.randn(1, 128, 16, 16)
        assert m(x).shape == (1, 128, 16, 16)

    def test_psablock_no_shortcut(self):
        m = PSABlock(64, num_heads=2, shortcut=False)
        x = torch.randn(1, 64, 8, 8)
        assert m(x).shape == (1, 64, 8, 8)


class TestC2PSALayer:
    def test_c2psa_basic(self):
        m = C2PSA(128, 128, n=1)
        x = torch.randn(1, 128, 20, 20)
        assert m(x).shape == (1, 128, 20, 20)

    def test_c2psa_multiple_blocks(self):
        m = C2PSA(64, 64, n=2)
        x = torch.randn(1, 64, 16, 16)
        assert m(x).shape == (1, 64, 16, 16)

    def test_c2psa_different_in_out(self):
        m = C2PSA(128, 64, n=1)
        x = torch.randn(1, 128, 10, 10)
        assert m(x).shape == (1, 64, 10, 10)


class TestPoolingLayers:
    def test_sppf(self):
        m = SPPF(64, 64)
        x = torch.randn(1, 64, 32, 32)
        assert m(x).shape == (1, 64, 32, 32)

    def test_spp(self):
        m = SPP(64, 64)
        x = torch.randn(1, 64, 32, 32)
        assert m(x).shape == (1, 64, 32, 32)


class TestAttentionLayers:
    def test_se_block(self):
        m = SEBlock(64)
        x = torch.randn(1, 64, 16, 16)
        assert m(x).shape == x.shape

    def test_cbam(self):
        m = CBAM(64)
        x = torch.randn(1, 64, 16, 16)
        assert m(x).shape == x.shape

    def test_transformer_block(self):
        m = TransformerBlock(64, num_heads=4)
        x = torch.randn(1, 64, 8, 8)
        assert m(x).shape == x.shape


class TestPositionalEncoding:
    def test_2d_encoding(self):
        pe = PositionalEncoding2D(256)
        x = torch.randn(1, 256, 8, 8)
        out = pe(x)
        assert out.shape == (1, 256, 8, 8)


class TestDeformableAttention:
    def test_forward(self):
        attn = DeformableAttention(d_model=64, n_heads=4, n_points=4)
        query = torch.randn(1, 64, 64)
        value = torch.randn(1, 64, 64)
        out = attn(query, value, (8, 8))
        assert out.shape == (1, 64, 64)


class TestMLP:
    def test_output_shape(self):
        mlp = MLP(in_dim=256, hidden_dim=256, out_dim=4, num_layers=3)
        x = torch.randn(2, 10, 256)
        out = mlp(x)
        assert out.shape == (2, 10, 4)

    def test_single_layer(self):
        mlp = MLP(in_dim=64, hidden_dim=64, out_dim=32, num_layers=1)
        x = torch.randn(1, 64)
        out = mlp(x)
        assert out.shape == (1, 32)
