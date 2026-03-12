"""
可变形注意力模块 — 用于 RF-DETR 的高效多尺度注意力。

纯 PyTorch 实现，不依赖 CUDA 自定义算子。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableAttention(nn.Module):
    """多尺度可变形注意力的简化实现。

    每个 query 只关注少量采样点（由学习的偏移量决定），
    而非全局注意力，从而大幅降低计算复杂度。

    Parameters
    ----------
    d_model : int
        隐藏维度。
    n_heads : int
        注意力头数。
    n_points : int
        每个头的采样点数。
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8, n_points: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query: torch.Tensor, value: torch.Tensor,
                spatial_shape: tuple) -> torch.Tensor:
        """前向推理。

        Parameters
        ----------
        query : Tensor (B, N_q, d_model)
        value : Tensor (B, N_v, d_model)
        spatial_shape : tuple (H, W) — value 的空间尺寸

        Returns
        -------
        Tensor (B, N_q, d_model)
        """
        B, N_q, _ = query.shape
        H, W = spatial_shape

        value = self.value_proj(value)  # (B, HW, d_model)
        value = value.view(B, H, W, self.n_heads, self.head_dim)

        offsets = self.sampling_offsets(query)  # (B, N_q, n_heads * n_points * 2)
        offsets = offsets.view(B, N_q, self.n_heads, self.n_points, 2)
        offsets = offsets.tanh()  # normalize to [-1, 1]

        attn_weights = self.attention_weights(query)  # (B, N_q, n_heads * n_points)
        attn_weights = attn_weights.view(B, N_q, self.n_heads, self.n_points)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, N_q, n_heads, n_points)

        # 为每个 query 生成参考点（均匀分布在空间上）
        ref_y = torch.linspace(0.5, H - 0.5, N_q if N_q <= H * W else H,
                               device=query.device) / H * 2 - 1
        ref_x = torch.linspace(0.5, W - 0.5, N_q if N_q <= H * W else W,
                               device=query.device) / W * 2 - 1

        if N_q == H * W:
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=query.device),
                torch.linspace(-1, 1, W, device=query.device),
                indexing="ij",
            )
            ref_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # (HW, 2)
        else:
            ref_points = torch.zeros(N_q, 2, device=query.device)

        ref_points = ref_points.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, N_q, 1, 1, 2)
        sampling_locations = ref_points + offsets * 0.5  # (B, N_q, n_heads, n_points, 2)
        sampling_locations = sampling_locations.clamp(-1, 1)

        # grid_sample: value needs (B*n_heads, head_dim, H, W) format
        value_perm = value.permute(0, 3, 4, 1, 2).reshape(B * self.n_heads, self.head_dim, H, W)

        # sampling for each head and point
        output = torch.zeros(B, N_q, self.n_heads, self.head_dim, device=query.device)
        for h in range(self.n_heads):
            for p in range(self.n_points):
                grid = sampling_locations[:, :, h, p, :].view(B, N_q, 1, 2)
                sampled = F.grid_sample(
                    value_perm[h::self.n_heads], grid,
                    mode="bilinear", align_corners=False,
                )  # (B, head_dim, N_q, 1)
                w = attn_weights[:, :, h, p].unsqueeze(-1)  # (B, N_q, 1)
                output[:, :, h, :] += sampled.squeeze(-1).permute(0, 2, 1) * w

        output = output.reshape(B, N_q, self.d_model)
        return self.output_proj(output)
