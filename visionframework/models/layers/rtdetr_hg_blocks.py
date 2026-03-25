"""HGStem / HGBlock / RepC3 for RT-DETR HG."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rtdetr_hg_conv import HGConv, HGDWConv, HGLightConv, HGRepConv


class HGStem(nn.Module):
    def __init__(self, c1: int, cm: int, c2: int):
        super().__init__()
        self.stem1 = HGConv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = HGConv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = HGConv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = HGConv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = HGConv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    def __init__(
        self,
        c1: int,
        cm: int,
        c2: int,
        k: int = 3,
        n: int = 6,
        lightconv: bool = False,
        shortcut: bool = False,
        act: nn.Module = None,
    ):
        super().__init__()
        if act is None:
            act = nn.ReLU()
        block = HGLightConv if lightconv else HGConv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = HGConv(c1 + n * cm, c2 // 2, 1, 1, act=act)
        self.ec = HGConv(c2 // 2, c2, 1, 1, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class RepC3(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 3, e: float = 1.0):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = HGConv(c1, c_, 1, 1)
        self.cv2 = HGConv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[HGRepConv(c_, c_) for _ in range(n)])
        self.cv3 = HGConv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))
