"""
统一 YOLO Backbone — 基于 C3k2 + SPPF + C2PSA 的特征提取网络。

通过 depth/width 缩放实现 n/s/m/l/x 不同尺寸变体。
YOLO11 和 YOLO26 共享同一拓扑，差异仅在 neck/head 配置。
"""

import torch.nn as nn
from typing import List, Optional

from visionframework.core.registry import BACKBONES
from visionframework.layers import ConvBNAct, C3k2, SPPF, C2PSA


def _ch(base: int, width: float, max_ch: int = 1024) -> int:
    return min(max(1, int(base * width)), max_ch)


def _n(base: int, depth: float) -> int:
    return max(1, round(base * depth))


@BACKBONES.register("YOLOBackbone")
class YOLOBackbone(nn.Module):
    """统一 YOLO backbone: Conv → C3k2 → SPPF → C2PSA，输出 [P3, P4, P5]。

    Parameters
    ----------
    depth : float   — 深度缩放因子 (0.5=n/s, 1.0=l/x)
    width : float   — 宽度缩放因子 (0.25=n, 0.5=s, 1.0=m/l)
    max_channels : int — 通道上限 (1024 for n/s, 512 for m/l/x)
    in_channels : int  — 输入图像通道数
    c3k2_12_c3k : bool — when True, first two C3k2 blocks use C3k (cv1/cv2/cv3 + m) for YOLO26m checkpoint keys
    bottleneck_k : int — kernel size for C3k bottlenecks (e.g. 3 for YOLO26m)
    """

    def __init__(self, depth: float = 0.5, width: float = 0.25,
                 max_channels: int = 1024, in_channels: int = 3,
                 sppf_cv1_act: bool = True, sppf_residual: bool = False,
                 c3k2_12_c3k: bool = False, bottleneck_k: Optional[int] = None, **_kw):
        super().__init__()
        ch0 = _ch(64, width, max_channels)
        ch1 = _ch(128, width, max_channels)
        ch2 = _ch(256, width, max_channels)
        ch3 = _ch(512, width, max_channels)
        ch4 = _ch(1024, width, max_channels)

        n_c3k2 = _n(2, depth)
        n_c2psa = _n(2, depth)

        self.conv0 = ConvBNAct(in_channels, ch0, 3, 2)
        self.conv1 = ConvBNAct(ch0, ch1, 3, 2)
        # Ultralytics weights differ by size: for n/s first blocks often use k=3,
        # while m/l/x typically use k=1. We follow a simple width-based default,
        # and keep bottleneck_k configurable for special cases.
        # n/s: 前两块与后两块均用 3x3；YOLO26m/l: 全部 C3k 用 3x3（bottleneck_k=3）
        k12 = 3 if width < 1.0 else 1
        k12 = bottleneck_k if bottleneck_k is not None else k12
        k34 = 3 if width < 1.0 else 1
        k34 = bottleneck_k if bottleneck_k is not None else k34
        self.c3k2_1 = C3k2(ch1, ch2, n=n_c3k2, c3k=c3k2_12_c3k, e=0.25, bottleneck_k=k12)
        self.conv3 = ConvBNAct(ch2, ch2, 3, 2)
        self.c3k2_2 = C3k2(ch2, ch3, n=n_c3k2, c3k=c3k2_12_c3k, e=0.25, bottleneck_k=k12)
        self.conv5 = ConvBNAct(ch3, ch3, 3, 2)
        self.c3k2_3 = C3k2(ch3, ch3, n=n_c3k2, c3k=True, bottleneck_k=k34)
        self.conv7 = ConvBNAct(ch3, ch4, 3, 2)
        self.c3k2_4 = C3k2(ch4, ch4, n=n_c3k2, c3k=True, bottleneck_k=k34)
        self.sppf = SPPF(ch4, ch4, k=5, cv1_act=sppf_cv1_act, residual=sppf_residual)
        self.c2psa = C2PSA(ch4, ch4, n=n_c2psa)

        self.out_channels = [ch3, ch3, ch4]

    def forward(self, x) -> List:
        x = self.conv0(x)
        x = self.c3k2_1(self.conv1(x))
        p3 = self.c3k2_2(self.conv3(x))
        p4 = self.c3k2_3(self.conv5(p3))
        p5 = self.c2psa(self.sppf(self.c3k2_4(self.conv7(p4))))
        return [p3, p4, p5]
