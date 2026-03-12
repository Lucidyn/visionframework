"""
ReID embedding head.

Produces a normalised feature vector for appearance-based re-identification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from visionframework.core.registry import HEADS
from visionframework.layers import ConvBNAct


@HEADS.register("ReIDHead")
class ReIDHead(nn.Module):
    """Global-average-pool → FC → L2-normalised embedding.

    Parameters
    ----------
    in_channels : list[int] | int
        If a list (from multi-scale neck), only the last scale is used.
    embedding_dim : int
        Output embedding size.
    """

    def __init__(self, in_channels, embedding_dim: int = 512, **_kw):
        super().__init__()
        if isinstance(in_channels, (list, tuple)):
            in_channels = in_channels[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn = nn.BatchNorm1d(in_channels)
        self.fc = nn.Linear(in_channels, embedding_dim, bias=False)
        self.bn2 = nn.BatchNorm1d(embedding_dim)

    def forward(self, features):
        if isinstance(features, (list, tuple)):
            features = features[-1]
        x = self.pool(features).flatten(1)
        x = self.bn(x)
        x = self.fc(x)
        x = self.bn2(x)
        return F.normalize(x, p=2, dim=1)
