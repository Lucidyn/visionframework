"""
统一 YOLO Backbone — 基于 C3k2 + SPPF + C2PSA 的特征提取网络。

通过 depth/width 缩放实现 n/s/m/l/x 不同尺寸变体。
YOLO11 和 YOLO26 共享同一拓扑，差异仅在 neck/head 配置。
"""

import torch.nn as nn
from typing import List

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
    """

    def __init__(self, depth: float = 0.5, width: float = 0.25,
                 max_channels: int = 1024, in_channels: int = 3, **_kw):
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
        self.c3k2_1 = C3k2(ch1, ch2, n=n_c3k2, c3k=False, e=0.25)
        self.conv3 = ConvBNAct(ch2, ch2, 3, 2)
        self.c3k2_2 = C3k2(ch2, ch3, n=n_c3k2, c3k=False, e=0.25)
        self.conv5 = ConvBNAct(ch3, ch3, 3, 2)
        self.c3k2_3 = C3k2(ch3, ch3, n=n_c3k2, c3k=True)
        self.conv7 = ConvBNAct(ch3, ch4, 3, 2)
        self.c3k2_4 = C3k2(ch4, ch4, n=n_c3k2, c3k=True)
        self.sppf = SPPF(ch4, ch4, k=5)
        self.c2psa = C2PSA(ch4, ch4, n=n_c2psa)

        self.out_channels = [ch3, ch3, ch4]

    def forward(self, x) -> List:
        x = self.conv0(x)
        x = self.c3k2_1(self.conv1(x))
        p3 = self.c3k2_2(self.conv3(x))
        p4 = self.c3k2_3(self.conv5(p3))
        p5 = self.c2psa(self.sppf(self.c3k2_4(self.conv7(p4))))
        return [p3, p4, p5]
