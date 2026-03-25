"""Ultralytics-layout conv blocks for RT-DETR HG (checkpoint-compatible submodule names)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .rtdetr_hg_ops import autopad


class HGConv(nn.Module):
    """Conv2d + BN + act (default SiLU), matching Ultralytics ``Conv``."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        if act is True:
            self.act = self.default_act
        elif isinstance(act, nn.Module):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HGDWConv(HGConv):
    """Depthwise conv (Ultralytics ``DWConv``)."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        g = math.gcd(c1, c2)
        super().__init__(c1, c2, k, s, g=g, d=d, act=act)


class HGLightConv(nn.Module):
    """1x1 + depthwise (Ultralytics ``LightConv``)."""

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        super().__init__()
        self.conv1 = HGConv(c1, c2, 1, 1, act=False)
        self.conv2 = HGDWConv(c2, c2, k, 1, act=act)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class HGRepConv(nn.Module):
    """RepConv (Ultralytics layout for RT-DETR ``RepC3``)."""

    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.bn = nn.BatchNorm2d(num_features=c1, eps=0.001, momentum=0.03) if bn and c2 == c1 and s == 1 else None
        self.conv1 = HGConv(c1, c2, k, s, p=p, g=g, d=d, act=False)
        self.conv2 = HGConv(c1, c2, 1, s, p=(p - k // 2), g=g, d=d, act=False)

    def forward(self, x):
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)


class HGConcat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x: list):
        return torch.cat(x, self.d)
