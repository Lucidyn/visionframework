"""
ReID embedding algorithm: extracts normalised appearance vectors from crops.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from visionframework.core.registry import ALGORITHMS
from visionframework.data.detection import Detection
from visionframework.utils.device import resolve_device


@ALGORITHMS.register("Embedder")
class Embedder:
    """Extract appearance embeddings for bounding-box crops.

    Parameters
    ----------
    model : nn.Module
        Built ReID model (backbone → neck → ReIDHead).
    input_size : tuple[int, int]
        Crop resize target ``(height, width)``.
    device : str
        Target device.
    fp16 : bool
        Half-precision inference.
    """

    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, int] = (256, 128),
        device: str = "auto",
        fp16: bool = False,
        **_kw,
    ):
        self.input_size = input_size
        self.fp16 = fp16 and torch.cuda.is_available()
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        if self.fp16:
            self.model = self.model.half()

    def _crop_and_preprocess(self, img: np.ndarray, bbox: Tuple) -> torch.Tensor:
        h, w = img.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((*self.input_size, 3), dtype=np.uint8)
        crop = cv2.resize(crop, (self.input_size[1], self.input_size[0]))
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        if self.fp16:
            tensor = tensor.half()
        return tensor.to(self.device)

    @torch.no_grad()
    def extract(self, img: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Return ``(N, dim)`` normalised embeddings for each detection's crop."""
        if not detections:
            return np.empty((0, 0), dtype=np.float32)
        batch = torch.stack([self._crop_and_preprocess(img, d.bbox) for d in detections])
        embeddings = self.model(batch)
        return embeddings.cpu().numpy()
