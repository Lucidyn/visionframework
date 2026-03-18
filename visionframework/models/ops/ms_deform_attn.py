# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
#
# Pure-PyTorch fallback implementation, integrated into visionframework.
# ------------------------------------------------------------------------

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


def _ms_deform_attn_core_pytorch(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    b, l, n_heads, d_head = value.shape
    _, len_q, _, n_levels, n_points, _ = sampling_locations.shape

    value_list = []
    start = 0
    for (h, w) in spatial_shapes.tolist():
        len_l = h * w
        value_list.append(value[:, start : start + len_l])
        start += len_l

    outputs = []
    for lvl, (h, w) in enumerate(spatial_shapes.tolist()):
        v = value_list[lvl].permute(0, 2, 3, 1).contiguous().view(b * n_heads, d_head, h, w)

        grid = sampling_locations[:, :, :, lvl].contiguous()
        grid = grid.permute(0, 2, 1, 3, 4).contiguous().view(b * n_heads, len_q, n_points, 2)
        grid = grid * 2.0 - 1.0

        sampled = F.grid_sample(v, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.view(b, n_heads, d_head, len_q, n_points).permute(0, 3, 1, 4, 2).contiguous()

        attn = attention_weights[:, :, :, lvl].unsqueeze(-1)
        out = (sampled * attn).sum(dim=3)
        outputs.append(out)

    out = sum(outputs)
    return out.view(b, len_q, n_heads * d_head)


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        nn.init.constant_(self.sampling_offsets.bias.data, 0.0)
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, len_q, _ = query.shape
        b2, len_in, _ = input_flatten.shape
        assert b == b2

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], 0.0)

        value = value.view(b, len_in, self.n_heads, self.d_model // self.n_heads)

        sampling_offsets = self.sampling_offsets(query).view(
            b, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(b, len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(b, len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = input_spatial_shapes[..., [1, 0]].view(1, 1, 1, self.n_levels, 1, 2)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer
        else:
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * reference_points[
                :, :, None, :, None, 2:
            ] * 0.5

        out = _ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        out = self.output_proj(out)
        return out

