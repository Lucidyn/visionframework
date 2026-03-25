"""
MLP — 简单多层感知机，供 DETR 等 Transformer 检测头共享使用。
"""

import torch.nn as nn


class MLP(nn.Module):
    """多层感知机（全连接 + ReLU）。

    Parameters
    ----------
    in_dim : int
        输入维度。
    hidden_dim : int
        隐藏层维度。
    out_dim : int
        输出维度。
    num_layers : int
        总层数（含输入和输出层）。
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden_dim
            d_out = out_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(d_in, d_out))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
