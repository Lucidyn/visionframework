"""
BaseAlgorithm — 所有推理算法的公共基类。

提供设备管理、fp16 设置、predict_batch 等共享逻辑。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional

from visionframework.utils.device import resolve_device


class BaseAlgorithm:
    """所有推理算法的基类。

    子类只需实现 ``predict(img)`` 方法。

    Parameters
    ----------
    model : nn.Module
        已构建的模型。
    device : str
        ``'cpu'``、``'cuda'`` 或 ``'auto'``。
    fp16 : bool
        在 CUDA 上使用半精度推理。
    """

    def __init__(self, model: Optional[nn.Module], device: str = "auto",
                 fp16: bool = False, **_kw):
        self.fp16 = fp16 and torch.cuda.is_available()
        self.device = resolve_device(device)
        self.model = None
        if model is not None:
            self.model = model.to(self.device).eval()
            if self.fp16:
                self.model = self.model.half()

    def predict(self, img: np.ndarray):
        raise NotImplementedError

    def predict_batch(self, images: List[np.ndarray]) -> list:
        """对一批图像逐张调用 predict。"""
        return [self.predict(img) for img in images]
