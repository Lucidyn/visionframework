"""
Attention modules.
"""

import torch
import torch.nn as nn
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ChannelAttention(nn.Module):
    """CBAM channel sub-module."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        avg = x.mean(dim=[2, 3])
        mx = x.amax(dim=[2, 3])
        w = (self.mlp(avg) + self.mlp(mx)).sigmoid().view(b, c, 1, 1)
        return x * w


class SpatialAttention(nn.Module):
    """CBAM spatial sub-module."""

    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        desc = torch.cat([x.mean(dim=1, keepdim=True),
                          x.amax(dim=1, keepdim=True)], dim=1)
        return x * self.conv(desc).sigmoid()


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, channels, reduction=16, spatial_k=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_k)

    def forward(self, x):
        return self.sa(self.ca(x))


class TransformerBlock(nn.Module):
    """Single self-attention transformer block for feature refinement."""

    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, C, H, W) → flatten → attend → reshape
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)          # (B, HW, C)
        normed = self.norm1(tokens)
        tokens = tokens + self.attn(normed, normed, normed, need_weights=False)[0]
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).view(B, C, H, W)
