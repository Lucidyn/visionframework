"""
统一 YOLO PAN neck — 基于 C3k2 的路径聚合网络。

通过 ``c3k`` 参数控制是否使用 C3k 大核 Bottleneck：
- YOLO11: ``c3k=false``（仅最后一个 C3k2 使用 c3k=True）
- YOLO26: ``c3k=true``（全部使用 c3k=True）

通道配置与 ultralytics 保持对齐：
- out_channels 为 PAN 三层实际输出通道 [P3_out, P4_out, P5_out]
- 对于 YOLO11n: out_channels=[64, 128, 256]（P3_out=64 ≠ P3_backbone=128）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from visionframework.core.registry import NECKS
from visionframework.layers import ConvBNAct, C3k2


@NECKS.register("YOLOPAN")
class YOLOPAN(nn.Module):
    """统一 YOLO PAN neck.

    Parameters
    ----------
    in_channels : list[int]
        来自 backbone 的三尺度通道 [P3_ch, P4_ch, P5_ch]。
    out_channels : list[int], optional
        PAN 三层输出通道。默认与 in_channels 相同。
        对于 YOLO11n 应显式设为 [64, 128, 256]。
    depth : float
        深度缩放因子。
    c3k : bool
        若 True，所有 C3k2 块使用 c3k=True；否则仅最后一个。
    """

    def __init__(self, in_channels: List[int], out_channels: List[int] = None,
                 depth: float = 0.5, c3k: bool = False, a2c2f: bool = False,
                 **_kw):
        super().__init__()
        c3, c4, c5 = in_channels
        o3, o4, o5 = out_channels if out_channels else in_channels
        n = max(1, round(2 * depth))

        # top-down
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.td_c3k2_p4 = C3k2(c4 + c5, o4, n=n, c3k=c3k)
        self.td_c3k2_p3 = C3k2(c3 + o4, o3, n=n, c3k=c3k)

        # bottom-up
        self.bu_down_p3 = ConvBNAct(o3, o3, 3, 2)
        self.bu_c3k2_p4 = C3k2(o3 + o4, o4, n=n, c3k=c3k)
        self.bu_down_p4 = ConvBNAct(o4, o4, 3, 2)
        self.bu_c3k2_p5 = C3k2(
            o4 + c5, o5, n=n if c3k else 1,
            c3k=not a2c2f and True,
            a2c2f=a2c2f,
        )

        self.out_channels = [o3, o4, o5]

    def forward(self, features: List) -> List:
        p3_in, p4_in, p5_in = features

        p4_td = self.td_c3k2_p4(torch.cat([
            F.interpolate(p5_in, size=p4_in.shape[2:], mode="nearest"), p4_in
        ], dim=1))
        p3_td = self.td_c3k2_p3(torch.cat([
            F.interpolate(p4_td, size=p3_in.shape[2:], mode="nearest"), p3_in
        ], dim=1))

        p4_out = self.bu_c3k2_p4(torch.cat([self.bu_down_p3(p3_td), p4_td], dim=1))
        p5_out = self.bu_c3k2_p5(torch.cat([self.bu_down_p4(p4_out), p5_in], dim=1))

        return [p3_td, p4_out, p5_out]
