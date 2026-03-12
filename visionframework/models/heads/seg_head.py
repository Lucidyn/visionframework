"""
Semantic / instance segmentation head.

Operates on multi-scale features from a neck and produces per-pixel
class logits at 1/4 input resolution (by default).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from visionframework.core.registry import HEADS
from visionframework.layers import ConvBNAct


@HEADS.register("SegHead")
class SegHead(nn.Module):
    """Simple FPN-style segmentation head.

    Parameters
    ----------
    in_channels : list[int]
        Channel counts per FPN level.
    num_classes : int
        Number of segmentation classes.
    hidden_dim : int
        Hidden channel dimension after fusion.
    """

    def __init__(self, in_channels: List[int], num_classes: int = 21,
                 hidden_dim: int = 256, **_kw):
        super().__init__()
        self.num_classes = num_classes
        self.lateral_convs = nn.ModuleList([
            ConvBNAct(c, hidden_dim, 1, 1) for c in in_channels
        ])
        self.fuse = ConvBNAct(hidden_dim * len(in_channels), hidden_dim, 3, 1)
        self.cls_pred = nn.Conv2d(hidden_dim, num_classes, 1)

    def forward(self, features: List):
        target_size = features[0].shape[2:]
        aligned = []
        for lat, feat in zip(self.lateral_convs, features):
            out = lat(feat)
            if out.shape[2:] != target_size:
                out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
            aligned.append(out)
        fused = self.fuse(torch.cat(aligned, dim=1))
        return self.cls_pred(fused)
