"""
DETR Head — 与 Facebook DETR 官方实现对齐。

Decoder 将 pos 只加到 Q/K（不加到 V），
self-attention 中 query_pos 加到 Q/K，
cross-attention 中 query_pos 加到 Q、memory_pos 加到 K。
"""

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from visionframework.core.registry import HEADS


class MLP(nn.Module):
    """简单多层感知机。"""

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


class DETRDecoderLayer(nn.Module):
    """DETR decoder 层 — 与官方 forward_post 对齐。"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        # self-attention: query_pos 加到 Q/K
        q = k = tgt if query_pos is None else tgt + query_pos
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # cross-attention: query_pos 加到 Q，pos 加到 K
        q = tgt if query_pos is None else tgt + query_pos
        k = memory if pos is None else memory + pos
        tgt2 = self.multihead_attn(query=q, key=k, value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # FFN
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class DETRDecoder(nn.Module):
    """DETR decoder — 堆叠 + 最后一层 norm。"""

    def __init__(self, decoder_layer: nn.Module, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(decoder_layer.norm1.normalized_shape[0])

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                pos: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)
        return self.norm(output)


@HEADS.register("DETRHead")
class DETRHead(nn.Module):
    """DETR 检测头（与官方实现对齐）。

    Parameters
    ----------
    d_model : int
        Transformer 隐藏维度。
    nhead : int
        Decoder 注意力头数。
    num_layers : int
        Decoder 层数。
    num_queries : int
        Object query 数量（最大检测数）。
    num_classes : int
        类别数（官方 COCO 用 91）。
    dim_feedforward : int
        Decoder FFN 维度。
    """

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, num_queries: int = 100,
                 num_classes: int = 91, dim_feedforward: int = 2048,
                 dropout: float = 0.1, in_channels: List[int] = None, **_kw):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.d_model = d_model

        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = DETRDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = DETRDecoder(decoder_layer, num_layers)

        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.bbox_head = MLP(d_model, d_model, 4, num_layers=3)

    def forward(self, neck_output) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向推理。

        Parameters
        ----------
        neck_output : tuple
            ``(memory, pos, (H, W))`` — encoder 输出 + 位置编码。

        Returns
        -------
        cls_logits : Tensor (B, num_queries, num_classes+1)
        bbox_pred : Tensor (B, num_queries, 4) — 归一化 cx, cy, w, h
        """
        memory, pos, (H, W) = neck_output
        B = memory.shape[0]

        # 转为 seq_len first: (N, B, d_model)
        memory_t = memory.permute(1, 0, 2)      # (HW, B, d_model)
        pos_t = pos.permute(1, 0, 2)            # (HW, B, d_model)
        query_embed = self.query_embed.weight.unsqueeze(1).expand(-1, B, -1)  # (Q, B, d_model)
        tgt = torch.zeros_like(query_embed)

        hs = self.decoder(tgt, memory_t, pos=pos_t, query_pos=query_embed)  # (Q, B, d_model)
        hs = hs.permute(1, 0, 2)  # (B, Q, d_model)

        cls_logits = self.class_head(hs)
        bbox_pred = self.bbox_head(hs).sigmoid()

        return cls_logits, bbox_pred
