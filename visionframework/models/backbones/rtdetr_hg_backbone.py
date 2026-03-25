"""RT-DETR HGNet backbone + hybrid encoder neck (``rtdetr-l`` / ``rtdetr-x``), pure PyTorch."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from visionframework.core.registry import BACKBONES

from .rtdetr_hg_encoder import build_rtdetr_hg_encoder


@BACKBONES.register("RTDETRHGBackbone")
class RTDETRHGBackbone(nn.Module):
    """HGStem + HGBlock + DWConv + PAN/AIFI/RepC3 stack up to (but not including) RTDETRDecoder."""

    def __init__(self, variant: str = "l", **_kw):
        super().__init__()
        variant = str(variant).lower()
        if variant not in ("l", "x"):
            raise ValueError("RTDETRHGBackbone variant must be 'l' or 'x'")
        layers, routing, out_idx = build_rtdetr_hg_encoder(variant)
        self.layers = layers
        self._routing: List[int | List[int]] = routing
        self._out_idx: Tuple[int, int, int] = out_idx
        self.variant = variant

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        y: List[torch.Tensor] = []
        for li, m in enumerate(self.layers):
            link = self._routing[li]
            if link != -1:
                if isinstance(link, int):
                    x = y[link]
                else:
                    x = [x if j == -1 else y[j] for j in link]
            x = m(x)
            y.append(x)
        return [y[i] for i in self._out_idx]
