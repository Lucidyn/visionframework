"""
设备解析工具 — 统一 auto/cpu/cuda 设备选择逻辑。
"""

import torch


def resolve_device(device: str = "auto") -> torch.device:
    """将 ``'auto'`` / ``'cpu'`` / ``'cuda'`` 等字符串解析为 ``torch.device``。"""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
