"""
DINOv2 Backbone — 基于 Vision Transformer 的特征提取。

通过 torch.hub 加载 Meta AI 的 DINOv2 预训练 ViT 模型，
提取多尺度特征供检测器等下游任务使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from visionframework.core.registry import BACKBONES


@BACKBONES.register("DINOv2Backbone")
class DINOv2Backbone(nn.Module):
    """DINOv2 ViT backbone。

    Parameters
    ----------
    model_name : str
        DINOv2 变体名: ``'dinov2_vits14'``, ``'dinov2_vitb14'``,
        ``'dinov2_vitl14'``, ``'dinov2_vitg14'``。
    out_dim : int
        输出投影维度（用于下游 neck/head 的统一通道数）。
    pretrained : bool
        是否加载预训练权重。
    freeze : bool
        是否冻结 backbone 参数。
    """

    _DIM_MAP = {
        "dinov2_vits14": 384,
        "dinov2_vitb14": 768,
        "dinov2_vitl14": 1024,
        "dinov2_vitg14": 1536,
    }

    def __init__(self, model_name: str = "dinov2_vitb14", out_dim: int = 256,
                 pretrained: bool = True, freeze: bool = False, **_kw):
        super().__init__()
        self.model_name = model_name
        self.patch_size = 14
        embed_dim = self._DIM_MAP.get(model_name, 768)

        if pretrained:
            self.vit = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        else:
            self.vit = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=False)

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.proj = nn.Conv2d(embed_dim, out_dim, 1)
        self.out_channels = [out_dim]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, C, H, W = x.shape
        features = self.vit.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]  # (B, N, D)

        h = H // self.patch_size
        w = W // self.patch_size
        feat_map = patch_tokens.transpose(1, 2).view(B, -1, h, w)
        feat_map = self.proj(feat_map)
        return [feat_map]
