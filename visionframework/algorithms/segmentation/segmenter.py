"""
Segmentation algorithm: wraps a segmentation model with pre/post-processing.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

from visionframework.core.registry import ALGORITHMS
from visionframework.algorithms.base import BaseAlgorithm


@ALGORITHMS.register("Segmenter")
class Segmenter(BaseAlgorithm):
    """Semantic segmentation algorithm.

    Parameters
    ----------
    model : nn.Module
        Built segmentation model (backbone → neck → SegHead).
    input_size : tuple[int, int]
        ``(height, width)`` the network expects.
    num_classes : int
        Number of segmentation categories.
    device : str
        Target device.
    fp16 : bool
        Half-precision inference.
    """

    def __init__(
        self,
        model,
        input_size: Tuple[int, int] = (640, 640),
        num_classes: int = 21,
        device: str = "auto",
        fp16: bool = False,
        **_kw,
    ):
        super().__init__(model=model, device=device, fp16=fp16)
        self.input_size = input_size
        self.num_classes = num_classes

    def _preprocess(self, img: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(img, (self.input_size[1], self.input_size[0]))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        if self.fp16:
            tensor = tensor.half()
        return tensor

    @torch.no_grad()
    def predict(self, img: np.ndarray) -> np.ndarray:
        """Return per-pixel class-id map at original resolution."""
        h0, w0 = img.shape[:2]
        tensor = self._preprocess(img)
        logits = self.model(tensor)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        seg_map = logits.argmax(dim=1)
        seg_map = F.interpolate(
            seg_map.unsqueeze(1).float(), size=(h0, w0), mode="nearest"
        ).squeeze().byte().cpu().numpy()
        return seg_map

