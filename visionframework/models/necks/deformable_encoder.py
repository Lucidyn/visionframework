"""
Deformable Encoder Neck — 用于 RF-DETR 的可变形注意力编码器。
"""

import torch
import torch.nn as nn
from typing import List

from visionframework.core.registry import NECKS
from visionframework.layers.deformable_attn import DeformableAttention
from visionframework.layers.positional import PositionalEncoding2D


class DeformableEncoderLayer(nn.Module):
    """单层可变形 encoder：self-attention + FFN。"""

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_points: int = 4, dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = DeformableAttention(d_model, n_heads, n_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, spatial_shape: tuple) -> torch.Tensor:
        src2 = self.self_attn(src, src, spatial_shape)
        src = self.norm1(src + src2)
        src = self.norm2(src + self.ffn(src))
        return src


@NECKS.register("DeformableEncoderNeck")
class DeformableEncoderNeck(nn.Module):
    """RF-DETR 风格的可变形注意力编码器 neck。

    Parameters
    ----------
    in_channels : list[int]
        backbone 输出通道。只使用最后一级。
    d_model : int
        隐藏维度。
    nhead : int
        注意力头数。
    num_layers : int
        Encoder 层数。
    n_points : int
        每头采样点数。
    dim_feedforward : int
        FFN 维度。
    """

    def __init__(self, in_channels: List[int], d_model: int = 256,
                 nhead: int = 8, num_layers: int = 6, n_points: int = 4,
                 dim_feedforward: int = 1024, dropout: float = 0.1, **_kw):
        super().__init__()
        c_in = in_channels[-1]
        self.input_proj = nn.Conv2d(c_in, d_model, 1)
        self.pos_enc = PositionalEncoding2D(d_model)
        self.layers = nn.ModuleList([
            DeformableEncoderLayer(d_model, nhead, n_points, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.d_model = d_model
        self.out_channels = [d_model]

    def forward(self, features: List) -> tuple:
        src = features[-1]
        B, _, H, W = src.shape
        src = self.input_proj(src)
        pos = self.pos_enc(src)

        src_flat = src.flatten(2).transpose(1, 2)  # (B, HW, d_model)
        pos_flat = pos.flatten(2).transpose(1, 2)

        memory = src_flat + pos_flat
        for layer in self.layers:
            memory = layer(memory, (H, W))

        return memory, (H, W)
