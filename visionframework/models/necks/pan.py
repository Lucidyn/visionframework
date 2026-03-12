"""
Path Aggregation Network (PAN) neck — YOLOv8-style.

Top-down FPN fusion followed by bottom-up aggregation.
Input and output are both ``[P3, P4, P5]``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from visionframework.core.registry import NECKS
from visionframework.layers import ConvBNAct, C2f


@NECKS.register("PAN")
class PAN(nn.Module):
    """YOLOv8-style PAN neck.

    Parameters
    ----------
    in_channels : list[int]
        Channel counts from the backbone for each scale, e.g. ``[256, 512, 1024]``.
    depth : float
        Depth multiplier for C2f repeats.
    width : float
        Width multiplier (applied to in_channels if > 0, else identity).
    """

    def __init__(self, in_channels: List[int], depth: float = 1.0,
                 width: float = 0.0, **_kw):
        super().__init__()
        c3, c4, c5 = in_channels
        n = max(1, round(3 * depth))

        # --- top-down (P5 → P4 → P3) ---
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.td_c2f_p4 = C2f(c4 + c5, c4, n=n, shortcut=False)
        self.td_c2f_p3 = C2f(c3 + c4, c3, n=n, shortcut=False)

        # --- bottom-up (P3 → P4 → P5) ---
        self.bu_down_p3 = ConvBNAct(c3, c3, 3, 2)
        self.bu_c2f_p4  = C2f(c3 + c4, c4, n=n, shortcut=False)
        self.bu_down_p4 = ConvBNAct(c4, c4, 3, 2)
        self.bu_c2f_p5  = C2f(c4 + c5, c5, n=n, shortcut=False)

        self.out_channels = [c3, c4, c5]

    def forward(self, features: List) -> List:
        p3_in, p4_in, p5_in = features

        # top-down
        p4_td = self.td_c2f_p4(torch.cat([
            F.interpolate(p5_in, size=p4_in.shape[2:], mode="nearest"), p4_in
        ], dim=1))
        p3_td = self.td_c2f_p3(torch.cat([
            F.interpolate(p4_td, size=p3_in.shape[2:], mode="nearest"), p3_in
        ], dim=1))

        # bottom-up
        p4_out = self.bu_c2f_p4(torch.cat([self.bu_down_p3(p3_td), p4_td], dim=1))
        p5_out = self.bu_c2f_p5(torch.cat([self.bu_down_p4(p4_out), p5_in], dim=1))

        return [p3_td, p4_out, p5_out]
