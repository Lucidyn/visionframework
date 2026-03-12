"""
DETR 检测算法 — 无 NMS 的集合预测。

预处理使用 ImageNet 归一化（与 Facebook DETR 官方一致）。
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple

from visionframework.core.registry import ALGORITHMS
from visionframework.data.detection import Detection
from visionframework.utils.filter import resolve_filter_ids
from visionframework.algorithms.base import BaseAlgorithm

# ImageNet 归一化参数
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@ALGORITHMS.register("DETRDetector")
class DETRDetector(BaseAlgorithm):
    """DETR-style 检测器（与 Facebook DETR 官方预处理对齐）。

    Parameters
    ----------
    model : nn.Module
        ModelWrapper (backbone → TransformerEncoderNeck → DETRHead)。
    input_size : tuple[int, int]
        输入尺寸 (H, W)。最长边会 resize 到这个值。
    conf : float
        置信度阈值。
    class_names : list[str] | None
        类别名映射。
    device : str
        目标设备。
    fp16 : bool
        半精度推理。
    """

    def __init__(
        self,
        model,
        input_size: Tuple[int, int] = (800, 800),
        conf: float = 0.7,
        class_names: Optional[List[str]] = None,
        filter_classes: Optional[List] = None,
        device: str = "auto",
        fp16: bool = False,
        **_kw,
    ):
        super().__init__(model=model, device=device, fp16=fp16)
        self.input_size = input_size
        self.conf = conf
        self.class_names = class_names
        self._filter_ids = resolve_filter_ids(filter_classes, class_names)

    def _preprocess(self, img: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """DETR 预处理: resize（保持比例）→ ImageNet normalize → tensor。"""
        h0, w0 = img.shape[:2]
        max_size = max(self.input_size)
        scale = min(max_size / max(h0, w0), max_size / min(h0, w0))
        nh, nw = int(h0 * scale), int(w0 * scale)
        resized = cv2.resize(img, (nw, nh))

        # BGR → RGB → float [0,1] → ImageNet normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        if self.fp16:
            tensor = tensor.half()
        return tensor, (h0, w0)

    @torch.no_grad()
    def predict(self, img: np.ndarray) -> List[Detection]:
        tensor, (h0, w0) = self._preprocess(img)

        cls_logits, bbox_pred = self.model(tensor)
        # cls_logits: (B, Q, nc+1), bbox_pred: (B, Q, 4) — cx, cy, w, h normalized

        probs = cls_logits[0].softmax(dim=-1)
        scores, cls_ids = probs[:, :-1].max(dim=-1)

        keep = scores > self.conf
        scores = scores[keep]
        cls_ids = cls_ids[keep]
        boxes = bbox_pred[0][keep]

        if self._filter_ids is not None:
            cls_mask = torch.tensor(
                [int(c) in self._filter_ids for c in cls_ids],
                dtype=torch.bool, device=boxes.device,
            )
            scores = scores[cls_mask]
            cls_ids = cls_ids[cls_mask]
            boxes = boxes[cls_mask]

        # cx, cy, w, h → xyxy，映射到原图尺寸
        cx, cy, bw, bh = boxes.unbind(dim=-1)
        x1 = (cx - bw / 2) * w0
        y1 = (cy - bh / 2) * h0
        x2 = (cx + bw / 2) * w0
        y2 = (cy + bh / 2) * h0

        detections = []
        for i in range(len(scores)):
            cid = int(cls_ids[i])
            detections.append(Detection(
                bbox=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                confidence=float(scores[i]),
                class_id=cid,
                class_name=self.class_names[cid] if self.class_names and cid < len(self.class_names) else None,
            ))
        return detections
