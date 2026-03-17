"""
CSP (Cross Stage Partial) building blocks.

包含 YOLOv5/v8 经典模块以及 YOLO11/YOLO26 的新模块 (C3k2, C2PSA)。
"""

import torch
import torch.nn as nn
from .conv import ConvBNAct


class Bottleneck(nn.Module):
    """Standard bottleneck: 1x1 → 3x3, with optional residual."""

    def __init__(self, c_in, c_out, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = ConvBNAct(c_in, c_hidden, k[0], 1)
        self.cv2 = ConvBNAct(c_hidden, c_out, k[1], 1, g=g)
        self.shortcut = shortcut and c_in == c_out

    def forward(self, x):
        out = self.cv2(self.cv1(x))
        return x + out if self.shortcut else out


class CSPBlock(nn.Module):
    """CSP Bottleneck block (YOLOv5-style)."""

    def __init__(self, c_in, c_out, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = ConvBNAct(c_in, c_hidden, 1, 1)
        self.cv2 = ConvBNAct(c_in, c_hidden, 1, 1)
        self.cv3 = ConvBNAct(2 * c_hidden, c_out, 1, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_hidden, c_hidden, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class C2f(nn.Module):
    """YOLOv8-style C2f block — faster CSP with split/concat."""

    def __init__(self, c_in, c_out, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c_out * e)
        self.cv1 = ConvBNAct(c_in, 2 * self.c, 1, 1)
        self.cv2 = ConvBNAct((2 + n) * self.c, c_out, 1, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


# ---------------------------------------------------------------------------
# YOLO11 / YOLO26 新增模块
# ---------------------------------------------------------------------------

class C3k(nn.Module):
    """C3-style block with configurable kernel size for inner bottlenecks."""

    def __init__(self, c_in, c_out, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_hidden = int(c_out * e)
        self.cv1 = ConvBNAct(c_in, c_hidden, 1, 1)
        self.cv2 = ConvBNAct(c_in, c_hidden, 1, 1)
        self.cv3 = ConvBNAct(2 * c_hidden, c_out, 1, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_hidden, c_hidden, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], dim=1))


class C3k2(nn.Module):
    """YOLO11 核心模块 — C2f 变体，使用两个小卷积替代单个大卷积。

    当 ``c3k=True`` 时，内部使用 C3k（可配置 kernel）替代普通 Bottleneck，
    扩大感受野，适合更深层的特征提取。

    Parameters
    ----------
    c_in : int   — 输入通道数
    c_out : int  — 输出通道数
    n : int      — Bottleneck 重复次数（会被 depth 缩放）
    c3k : bool   — 是否使用 C3k 内部模块
    e : float    — 通道扩展比例
    """

    def __init__(self, c_in, c_out, n=1, c3k=False, a2c2f=False,
                 e=0.5, g=1, shortcut=True, bottleneck_k=3):
        super().__init__()
        self.c = int(c_out * e)
        self.cv1 = ConvBNAct(c_in, 2 * self.c, 1, 1)
        self.cv2 = ConvBNAct((2 + n) * self.c, c_out, 1, 1)

        if a2c2f:
            self.m = nn.ModuleList(
                nn.Sequential(
                    Bottleneck(self.c, self.c, shortcut=shortcut, g=g),
                    PSABlock(self.c, attn_ratio=0.5,
                             num_heads=max(self.c // 64, 1), shortcut=shortcut),
                ) for _ in range(n)
            )
        elif c3k:
            self.m = nn.ModuleList(
                C3k(self.c, self.c, n=2, shortcut=shortcut, k=bottleneck_k) for _ in range(n)
            )
        else:
            self.m = nn.ModuleList(
                Bottleneck(self.c, self.c, shortcut=shortcut, g=g, k=(bottleneck_k, bottleneck_k))
                for _ in range(n)
            )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class Attention(nn.Module):
    """Conv-based multi-head self-attention（与 ultralytics 结构对齐）。

    使用 1×1 Conv 生成 QKV，depthwise 3×3 Conv 注入位置编码。
    全程在 4D 空间张量上操作，无需 flatten/reshape。
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = ConvBNAct(dim, h, 1, act=False)
        self.proj = ConvBNAct(dim, dim, 1, act=False)
        self.pe = ConvBNAct(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class PSABlock(nn.Module):
    """Position-Sensitive Attention block（与 ultralytics 结构对齐）。

    在 4D 空间张量上执行 Attention + FFN，带可选残差连接。
    """

    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4,
                 shortcut: bool = True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(
            ConvBNAct(c, c * 2, 1),
            ConvBNAct(c * 2, c, 1, act=False),
        )
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """C2PSA 模块 — CSP 结构 + Position-Sensitive Attention（与 ultralytics 对齐）。

    将输入分为 a（shortcut）和 b（通过 PSABlock 堆叠），拼接后融合。
    """

    def __init__(self, c_in, c_out=None, n=1, e=0.5):
        super().__init__()
        c_out = c_out or c_in
        self.c = int(c_in * e)
        self.cv1 = ConvBNAct(c_in, 2 * self.c, 1, 1)
        self.cv2 = ConvBNAct(2 * self.c, c_out, 1)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
              for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))
