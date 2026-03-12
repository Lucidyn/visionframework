"""
RF-DETR Head — 可变形 Transformer decoder + 预测 MLP。

与标准 DETRHead 类似，但 decoder 使用可变形交叉注意力
而非全局交叉注意力，提高效率。
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from visionframework.core.registry import HEADS
from visionframework.layers.deformable_attn import DeformableAttention


class RFDETRDecoderLayer(nn.Module):
    """RF-DETR decoder 层：self-attn + deformable cross-attn + FFN。"""

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_points: int = 4, dim_feedforward: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = DeformableAttention(d_model, n_heads, n_points)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                spatial_shape: tuple) -> torch.Tensor:
        # self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt, need_weights=False)[0]
        tgt = self.norm1(tgt + tgt2)
        # deformable cross-attention
        tgt2 = self.cross_attn(tgt, memory, spatial_shape)
        tgt = self.norm2(tgt + tgt2)
        # FFN
        tgt = self.norm3(tgt + self.ffn(tgt))
        return tgt


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            d_out = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(d_in, d_out))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@HEADS.register("RFDETRHead")
class RFDETRHead(nn.Module):
    """RF-DETR 检测头。

    Parameters
    ----------
    d_model : int
        隐藏维度。
    nhead : int
        Decoder 注意力头数。
    num_layers : int
        Decoder 层数。
    num_queries : int
        Object query 数量。
    num_classes : int
        类别数。
    n_points : int
        可变形注意力采样点数。
    dim_feedforward : int
        FFN 维度。
    """

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, num_queries: int = 300,
                 num_classes: int = 80, n_points: int = 4,
                 dim_feedforward: int = 1024, dropout: float = 0.1,
                 in_channels: List[int] = None, **_kw):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.decoder_layers = nn.ModuleList([
            RFDETRDecoderLayer(d_model, nhead, n_points, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.bbox_head = MLP(d_model, d_model, 4, num_layers=3)

    def forward(self, neck_output) -> Tuple[torch.Tensor, torch.Tensor]:
        memory, (H, W) = neck_output
        B = memory.shape[0]

        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        hs = queries
        for layer in self.decoder_layers:
            hs = layer(hs, memory, (H, W))

        cls_logits = self.class_head(hs)
        bbox_pred = self.bbox_head(hs).sigmoid()

        return cls_logits, bbox_pred
