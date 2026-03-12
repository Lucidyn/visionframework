"""
位置编码模块 — 与 Facebook DETR 官方实现对齐的 2D 正弦位置编码。
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding2D(nn.Module):
    """2D 正弦/余弦位置编码（与 Facebook DETR 官方实现对齐）。

    使用 cumsum + normalize 将坐标映射到 [0, 2π]，
    然后按 sin/cos 交替编码 y 和 x 坐标。
    """

    def __init__(self, d_model: int, temperature: float = 10000.0,
                 normalize: bool = True, scale: float = None):
        super().__init__()
        self.num_pos_feats = d_model // 2
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B, C, H, W)，返回 (B, C, H, W) 位置编码。"""
        B, _, H, W = x.shape

        not_mask = torch.ones(B, H, W, dtype=torch.bool, device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # (B, H, W, C/2)
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                              pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                              pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (B, C, H, W)
        return pos
