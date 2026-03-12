"""
Feature Pyramid Network (FPN) neck.

Takes ``[P3, P4, P5]`` from a backbone and outputs fused features
at the same three scales via top-down lateral connections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from visionframework.core.registry import NECKS
from visionframework.layers import ConvBNAct


@NECKS.register("FPN")
class FPN(nn.Module):
    """Classic FPN: 1x1 lateral + 3x3 smooth at each level.

    Parameters
    ----------
    in_channels : list[int]
        Channel sizes from the backbone (e.g. [256, 512, 1024]).
    out_channels : int
        Uniform channel size for all output levels.
    """

    def __init__(self, in_channels: List[int], out_channels: int = 256, **_kw):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, out_channels, 1) for c in in_channels
        ])
        self.smooths = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels
        ])
        self.out_channels = [out_channels] * len(in_channels)

    def forward(self, features: List) -> List:
        lats = [lat(f) for lat, f in zip(self.laterals, features)]
        for i in range(len(lats) - 1, 0, -1):
            lats[i - 1] = lats[i - 1] + F.interpolate(
                lats[i], size=lats[i - 1].shape[2:], mode="nearest"
            )
        return [s(l) for s, l in zip(self.smooths, lats)]
