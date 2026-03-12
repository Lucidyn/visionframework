"""
Basic convolution building blocks.
"""

import torch
import torch.nn as nn
import math


def autopad(k, p=None, d=1):
    """Calculate same-padding size for a given kernel/dilation."""
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class ConvBNAct(nn.Module):
    """Conv2d → BatchNorm2d → Activation (SiLU by default)."""

    def __init__(self, c_in, c_out, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, autopad(k, p, d),
                              dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConvBNAct(nn.Module):
    """Depthwise Conv + BN + Act（与 ultralytics DWConv 对齐）。

    使用 ``groups=min(c_in, c_out)`` 实现真正的 depthwise 卷积。
    """

    def __init__(self, c_in, c_out, k=3, s=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, k // 2,
                              groups=min(c_in, c_out), bias=False)
        self.bn = nn.BatchNorm2d(c_out, eps=1e-3, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSepConv(nn.Module):
    """Depthwise-separable convolution: DW-Conv → PW-Conv."""

    def __init__(self, c_in, c_out, k=3, s=1, act=True):
        super().__init__()
        self.dw = ConvBNAct(c_in, c_in, k, s, g=c_in, act=act)
        self.pw = ConvBNAct(c_in, c_out, 1, 1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


class Focus(nn.Module):
    """Slice-and-concat to trade spatial resolution for channel depth."""

    def __init__(self, c_in, c_out, k=1, s=1, act=True):
        super().__init__()
        self.conv = ConvBNAct(c_in * 4, c_out, k, s, act=act)

    def forward(self, x):
        return self.conv(torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2],
        ], dim=1))
