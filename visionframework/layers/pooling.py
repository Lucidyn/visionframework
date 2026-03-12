"""
Spatial pooling layers.
"""

import torch
import torch.nn as nn
from .conv import ConvBNAct


class SPP(nn.Module):
    """Spatial Pyramid Pooling (multi-kernel max-pool)."""

    def __init__(self, c_in, c_out, k=(5, 9, 13)):
        super().__init__()
        c_hidden = c_in // 2
        self.cv1 = ConvBNAct(c_in, c_hidden, 1, 1)
        self.cv2 = ConvBNAct(c_hidden * (len(k) + 1), c_out, 1, 1)
        self.m = nn.ModuleList(
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in k
        )

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling – Fast (sequential 5x5 max-pool).

    Parameters
    ----------
    cv1_act : bool
        Whether to apply SiLU after cv1. YOLO11 uses True, YOLO26 uses False.
    residual : bool
        Whether to add input residual to output (YOLO26 uses True).
    """

    def __init__(self, c_in, c_out, k=5, cv1_act: bool = True, residual: bool = False):
        super().__init__()
        c_hidden = c_in // 2
        self.cv1 = ConvBNAct(c_in, c_hidden, 1, 1, act=cv1_act)
        self.cv2 = ConvBNAct(c_hidden * 4, c_out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.residual = residual

    def forward(self, x):
        inp = x
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        out = self.cv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))
        return out + inp if self.residual else out
