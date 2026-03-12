"""
ResNet backbone (18/34/50/101/152).

Produces multi-scale feature maps from layer2, layer3, layer4
(strides 8, 16, 32 by default).
"""

import torch
import torch.nn as nn
from typing import List

from visionframework.core.registry import BACKBONES


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c_in, c_out, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(c_out)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(c_out)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, c_in, c_out, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, stride, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(c_out)
        self.conv3 = nn.Conv2d(c_out, c_out * self.expansion, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(c_out * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


_CONFIGS = {
    18:  (BasicBlock,     [2, 2, 2, 2]),
    34:  (BasicBlock,     [3, 4, 6, 3]),
    50:  (BottleneckBlock, [3, 4, 6, 3]),
    101: (BottleneckBlock, [3, 4, 23, 3]),
    152: (BottleneckBlock, [3, 8, 36, 3]),
}


@BACKBONES.register("ResNet")
class ResNet(nn.Module):
    """ResNet backbone that outputs multi-scale features ``[C3, C4, C5]``.

    Parameters
    ----------
    layers : int
        Variant: 18, 34, 50, 101, or 152.
    in_channels : int
        Input image channels.
    """

    def __init__(self, layers: int = 50, in_channels: int = 3, **_kw):
        super().__init__()
        if layers not in _CONFIGS:
            raise ValueError(f"Unsupported ResNet variant: {layers}")
        block, counts = _CONFIGS[layers]

        self._inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.layer1 = self._make_layer(block, 64,  counts[0], stride=1)
        self.layer2 = self._make_layer(block, 128, counts[1], stride=2)
        self.layer3 = self._make_layer(block, 256, counts[2], stride=2)
        self.layer4 = self._make_layer(block, 512, counts[3], stride=2)

        self.out_channels = [
            128 * block.expansion,
            256 * block.expansion,
            512 * block.expansion,
        ]

    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self._inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self._inplanes, planes * block.expansion, 1, stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self._inplanes, planes, stride, downsample)]
        self._inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x) -> List:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        c3 = self.layer2(x)   # stride 8
        c4 = self.layer3(c3)  # stride 16
        c5 = self.layer4(c4)  # stride 32
        return [c3, c4, c5]
