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
    """Spatial Pyramid Pooling – Fast (sequential 5x5 max-pool)."""

    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        c_hidden = c_in // 2
        self.cv1 = ConvBNAct(c_in, c_hidden, 1, 1)
        self.cv2 = ConvBNAct(c_hidden * 4, c_out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))
