"""
CSPDarknet backbone (YOLOv5 / YOLOv8 style).

Produces multi-scale feature maps at strides 8, 16, 32.
Output is a *list* of feature tensors ``[P3, P4, P5]``.
"""

import torch.nn as nn
from typing import List

from visionframework.core.registry import BACKBONES
from visionframework.layers import ConvBNAct, C2f, SPPF


@BACKBONES.register("CSPDarknet")
class CSPDarknet(nn.Module):
    """YOLOv8-style CSPDarknet backbone.

    Parameters
    ----------
    depth : float
        Depth multiplier — scales the number of C2f repeats.
    width : float
        Width multiplier — scales channel counts.
    in_channels : int
        Input image channels (default 3).
    """

    _BASE_CHANNELS = [64, 128, 256, 512, 1024]
    _BASE_DEPTHS   = [3, 6, 6, 3]

    def __init__(self, depth: float = 1.0, width: float = 1.0, in_channels: int = 3, **_kw):
        super().__init__()
        ch = [max(1, int(c * width)) for c in self._BASE_CHANNELS]
        nd = [max(1, round(d * depth)) for d in self._BASE_DEPTHS]

        # stem
        self.stem = ConvBNAct(in_channels, ch[0], 3, 2)

        # stages (stride 4, 8, 16, 32)
        self.stage1 = nn.Sequential(ConvBNAct(ch[0], ch[1], 3, 2), C2f(ch[1], ch[1], n=nd[0], shortcut=True))
        self.stage2 = nn.Sequential(ConvBNAct(ch[1], ch[2], 3, 2), C2f(ch[2], ch[2], n=nd[1], shortcut=True))
        self.stage3 = nn.Sequential(ConvBNAct(ch[2], ch[3], 3, 2), C2f(ch[3], ch[3], n=nd[2], shortcut=True))
        self.stage4 = nn.Sequential(ConvBNAct(ch[3], ch[4], 3, 2), C2f(ch[4], ch[4], n=nd[3], shortcut=True), SPPF(ch[4], ch[4]))

        self.out_channels = [ch[2], ch[3], ch[4]]

    def forward(self, x) -> List:
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)   # stride 8
        p4 = self.stage3(p3)  # stride 16
        p5 = self.stage4(p4)  # stride 32
        return [p3, p4, p5]
