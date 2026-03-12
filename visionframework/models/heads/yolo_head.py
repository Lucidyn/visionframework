"""
Anchor-free YOLO detection head (ultralytics YOLO11/YOLO26-style).

Takes multi-scale features ``[P3, P4, P5]`` and produces
raw predictions: ``(batch, sum_of_grid_cells, 4 + num_classes)``.

Decoding into absolute box coordinates is handled by the Detector algorithm.
"""

import math
import torch
import torch.nn as nn
from typing import List

from visionframework.core.registry import HEADS
from visionframework.layers import ConvBNAct, DWConvBNAct


@HEADS.register("YOLOHead")
class YOLOHead(nn.Module):
    """Decoupled detection head aligned with ultralytics YOLO11/YOLO26.

    Reg branch: Conv(in_ch → c2, 3) → Conv(c2, c2, 3) → Conv2d(c2, 4*reg_max, 1)
    Cls branch: [DWConv(in_ch) + Conv1x1(→c3)] → [DWConv(c3) + Conv1x1(→c3)] → Conv2d(c3, nc, 1)

    Parameters
    ----------
    in_channels : list[int]
        Per-level input channel counts from the neck.
    num_classes : int
        Number of object categories.
    reg_max : int
        DFL register count for box regression.
    """

    def __init__(self, in_channels: List[int], num_classes: int = 80,
                 reg_max: int = 16, **_kw):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_levels = len(in_channels)

        c2 = max(16, in_channels[0] // 4, 4 * reg_max)
        c3 = max(in_channels[0], min(num_classes, 100))

        self.reg_convs = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_preds = nn.ModuleList()

        for ch in in_channels:
            self.reg_convs.append(nn.Sequential(
                ConvBNAct(ch, c2, 3, 1),
                ConvBNAct(c2, c2, 3, 1),
            ))
            self.cls_convs.append(nn.Sequential(
                nn.Sequential(DWConvBNAct(ch, ch, 3), ConvBNAct(ch, c3, 1, 1)),
                nn.Sequential(DWConvBNAct(c3, c3, 3), ConvBNAct(c3, c3, 1, 1)),
            ))
            self.reg_preds.append(nn.Conv2d(c2, 4 * reg_max, 1))
            self.cls_preds.append(nn.Conv2d(c3, num_classes, 1))

        self._initialize_biases()

    def _initialize_biases(self):
        for cls_pred in self.cls_preds:
            nn.init.constant_(cls_pred.bias, -math.log((1 - 0.01) / 0.01))
        for reg_pred in self.reg_preds:
            nn.init.constant_(reg_pred.bias, 1.0)

    def forward(self, features: List) -> List:
        """Return list of ``(cls_logits, reg_raw)`` per scale level."""
        outputs = []
        for i, feat in enumerate(features):
            cls_feat = self.cls_convs[i](feat)
            reg_feat = self.reg_convs[i](feat)
            cls_out = self.cls_preds[i](cls_feat)
            reg_out = self.reg_preds[i](reg_feat)
            outputs.append((cls_out, reg_out))
        return outputs
