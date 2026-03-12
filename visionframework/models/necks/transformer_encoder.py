"""
Transformer Encoder Neck — 与 Facebook DETR 官方实现对齐。

位置编码只加到 Q 和 K 上（不加到 V），与 PyTorch 标准 TransformerEncoder 不同。
"""

import copy
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from visionframework.core.registry import NECKS
from visionframework.layers.positional import PositionalEncoding2D


class DETREncoderLayer(nn.Module):
    """DETR encoder 层 — pos 只加到 Q/K。"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor,
                pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = k = src if pos is None else src + pos
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class DETREncoder(nn.Module):
    """DETR encoder — 堆叠多层 DETREncoderLayer。"""

    def __init__(self, encoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

    def forward(self, src: torch.Tensor,
                pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, pos=pos)
        return output


@NECKS.register("TransformerEncoderNeck")
class TransformerEncoderNeck(nn.Module):
    """DETR-style Transformer encoder neck（与官方实现对齐）。

    Parameters
    ----------
    in_channels : list[int]
        backbone 输出的多尺度通道。只使用最后一个尺度 (C5)。
    d_model : int
        Transformer 隐藏维度。
    nhead : int
        注意力头数。
    num_layers : int
        Encoder 层数。
    dim_feedforward : int
        FFN 内部维度。
    dropout : float
        Dropout 比例。
    """

    def __init__(self, in_channels: List[int], d_model: int = 256,
                 nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1, **_kw):
        super().__init__()
        c5 = in_channels[-1]
        self.input_proj = nn.Conv2d(c5, d_model, 1)
        self.pos_enc = PositionalEncoding2D(d_model, normalize=True)

        encoder_layer = DETREncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = DETREncoder(encoder_layer, num_layers)
        self.d_model = d_model
        self.out_channels = [d_model]

    def forward(self, features: List) -> tuple:
        src = features[-1]
        B, _, H, W = src.shape
        src = self.input_proj(src)       # (B, d_model, H, W)
        pos = self.pos_enc(src)          # (B, d_model, H, W)

        # seq_len first for nn.MultiheadAttention: (HW, B, d_model)
        src_flat = src.flatten(2).permute(2, 0, 1)
        pos_flat = pos.flatten(2).permute(2, 0, 1)

        memory = self.encoder(src_flat, pos=pos_flat)  # (HW, B, d_model)

        # 转回 batch_first: (B, HW, d_model)
        memory = memory.permute(1, 0, 2)
        pos_out = pos_flat.permute(1, 0, 2)  # (B, HW, d_model)
        return memory, pos_out, (H, W)
