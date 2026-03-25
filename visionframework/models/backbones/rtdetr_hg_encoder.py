"""Fixed RT-DETR HG encoder stacks (``l`` / ``x``) matching Ultralytics layer order and routing."""

from __future__ import annotations

from typing import List, Tuple

import torch.nn as nn

from visionframework.models.layers.rtdetr_hg_blocks import HGBlock, HGStem, RepC3
from visionframework.models.layers.rtdetr_hg_conv import HGConcat, HGConv, HGDWConv
from visionframework.models.layers.rtdetr_hg_transformer import AIFI


def build_rtdetr_hg_encoder(variant: str) -> Tuple[nn.ModuleList, List, Tuple[int, int, int]]:
    variant = variant.lower()
    if variant == "l":
        return _build_l()
    if variant == "x":
        return _build_x()
    raise ValueError("variant must be 'l' or 'x'")


def _build_l():
    """rtdetr-l: 28 layers before decoder; outputs at indices 21, 24, 27."""
    act_r = nn.ReLU(inplace=True)
    layers = nn.ModuleList()
    routing: List = []

    def add(m, f=-1):
        layers.append(m)
        routing.append(f)

    add(HGStem(3, 32, 48))
    add(HGBlock(48, 48, 128, 3, 6, False, False, act_r))
    add(HGDWConv(128, 128, 3, 2, act=False))
    add(HGBlock(128, 96, 512, 3, 6, False, False, act_r))
    add(HGDWConv(512, 512, 3, 2, act=False))
    add(HGBlock(512, 192, 1024, 5, 6, True, False, act_r))
    add(HGBlock(1024, 192, 1024, 5, 6, True, True, act_r))
    add(HGBlock(1024, 192, 1024, 5, 6, True, True, act_r))
    add(HGDWConv(1024, 1024, 3, 2, act=False))
    add(HGBlock(1024, 384, 2048, 5, 6, True, False, act_r))
    add(HGConv(2048, 256, 1, 1, act=False))
    add(AIFI(256, 1024, 8, dropout=0, act=nn.GELU()))
    add(HGConv(256, 256, 1, 1))
    add(nn.Upsample(scale_factor=2.0, mode="nearest"))
    add(HGConv(1024, 256, 1, 1, act=False), 7)
    add(HGConcat(1), [-2, -1])
    add(RepC3(512, 256, n=3))
    add(HGConv(256, 256, 1, 1))
    add(nn.Upsample(scale_factor=2.0, mode="nearest"))
    add(HGConv(512, 256, 1, 1, act=False), 3)
    add(HGConcat(1), [-2, -1])
    add(RepC3(512, 256, n=3))
    add(HGConv(256, 256, 3, 2))
    add(HGConcat(1), [-1, 17])
    add(RepC3(512, 256, n=3))
    add(HGConv(256, 256, 3, 2))
    add(HGConcat(1), [-1, 12])
    add(RepC3(512, 256, n=3))
    return layers, routing, (21, 24, 27)


def _build_x():
    """rtdetr-x: 32 layers before decoder; outputs at indices 25, 28, 31."""
    act_r = nn.ReLU(inplace=True)
    layers = nn.ModuleList()
    routing: List = []

    def add(m, f=-1):
        layers.append(m)
        routing.append(f)

    add(HGStem(3, 32, 64))
    add(HGBlock(64, 64, 128, 3, 6, False, False, act_r))
    add(HGDWConv(128, 128, 3, 2, act=False))
    add(HGBlock(128, 128, 512, 3, 6, False, False, act_r))
    add(HGBlock(512, 128, 512, 3, 6, False, True, act_r))
    add(HGDWConv(512, 512, 3, 2, act=False))
    add(HGBlock(512, 256, 1024, 5, 6, True, False, act_r))
    add(HGBlock(1024, 256, 1024, 5, 6, True, True, act_r))
    add(HGBlock(1024, 256, 1024, 5, 6, True, True, act_r))
    add(HGBlock(1024, 256, 1024, 5, 6, True, True, act_r))
    add(HGBlock(1024, 256, 1024, 5, 6, True, True, act_r))
    add(HGDWConv(1024, 1024, 3, 2, act=False))
    add(HGBlock(1024, 512, 2048, 5, 6, True, False, act_r))
    add(HGBlock(2048, 512, 2048, 5, 6, True, True, act_r))
    add(HGConv(2048, 384, 1, 1, act=False))
    add(AIFI(384, 2048, 8, dropout=0, act=nn.GELU()))
    add(HGConv(384, 384, 1, 1))
    add(nn.Upsample(scale_factor=2.0, mode="nearest"))
    add(HGConv(1024, 384, 1, 1, act=False), 10)
    add(HGConcat(1), [-2, -1])
    add(RepC3(768, 384, n=3))
    add(HGConv(384, 384, 1, 1))
    add(nn.Upsample(scale_factor=2.0, mode="nearest"))
    add(HGConv(512, 384, 1, 1, act=False), 4)
    add(HGConcat(1), [-2, -1])
    add(RepC3(768, 384, n=3))
    add(HGConv(384, 384, 3, 2))
    add(HGConcat(1), [-1, 21])
    add(RepC3(768, 384, n=3))
    add(HGConv(384, 384, 3, 2))
    add(HGConcat(1), [-1, 16])
    add(RepC3(768, 384, n=3))
    return layers, routing, (25, 28, 31)
