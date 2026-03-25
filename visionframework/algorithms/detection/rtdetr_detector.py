"""
RT-DETR 检测 — 预处理为 scale-fill 方形输入、输出按 xywh 再映射回原图（与 Ultralytics 官方 rtdetr-l/x 推理习惯一致）。
默认假定 ``numpy`` 图像为 **BGR**（与 ``cv2.imread`` 一致）；若为 RGB（如 PIL / ``plt.imread``），请设置 ``input_layout="rgb"``。
HGNet（l/x）权重请自备，经 ``convert_ultralytics_rtdetr_hg`` 转为框架格式；推理不依赖 ``ultralytics``（官方 ``.pt`` 许可见 NOTICE）。
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


def _xywh_to_xyxy_norm(cx: torch.Tensor, cy: torch.Tensor, w: torch.Tensor, h: torch.Tensor):
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return x1, y1, x2, y2


@ALGORITHMS.register("RTDETRDetector")
class RTDETRDetector(BaseAlgorithm):
    """RT-DETR（HGNet），模型为 ``ModelWrapper(RTDETRHGBackbone → RTDETRHGDecoder)``（无独立 neck 键）。

    Parameters
    ----------
    input_layout : str
        ``"bgr"``（默认，与 OpenCV 一致）或 ``"rgb"``。通道顺序错误会导致检测与可视化明显错位。
    """

    def __init__(
        self,
        model,
        input_size: int = 640,
        conf: float = 0.5,
        max_det: int = 300,
        class_names: Optional[List[str]] = None,
        filter_classes: Optional[List] = None,
        device: str = "auto",
        fp16: bool = False,
        input_layout: str = "bgr",
        **_kw,
    ):
        super().__init__(model=model, device=device, fp16=fp16)
        self.input_size = int(input_size)
        self.conf = float(conf)
        self.max_det = int(max_det)
        self.class_names = class_names
        self._filter_ids = resolve_filter_ids(filter_classes, class_names)
        layout = str(input_layout).lower()
        if layout not in ("bgr", "rgb"):
            raise ValueError("input_layout must be 'bgr' or 'rgb'")
        self._input_layout = layout
        self.model.eval()

    def _preprocess(self, img: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h0, w0 = img.shape[:2]
        resized = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        if self._input_layout == "rgb":
            rgb = resized.astype(np.float32) / 255.0
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        if self.fp16:
            tensor = tensor.half()
        return tensor, (h0, w0)

    @torch.no_grad()
    def predict(self, img: np.ndarray) -> List[Detection]:
        tensor, (h0, w0) = self._preprocess(img)
        raw = self.model(tensor)
        if isinstance(raw, (list, tuple)):
            raw = raw[0]
        # (1, Q, 4 + nc)
        pred = raw[0]
        boxes = pred[:, :4]
        scores = pred[:, 4:]
        max_score, cls_ids = scores.max(dim=-1)
        keep = max_score > self.conf
        boxes = boxes[keep]
        max_score = max_score[keep]
        cls_ids = cls_ids[keep]
        if self._filter_ids is not None:
            cls_mask = torch.tensor(
                [int(c) in self._filter_ids for c in cls_ids],
                dtype=torch.bool,
                device=boxes.device,
            )
            boxes = boxes[cls_mask]
            max_score = max_score[cls_mask]
            cls_ids = cls_ids[cls_mask]

        cx, cy, bw, bh = boxes.unbind(dim=-1)
        x1, y1, x2, y2 = _xywh_to_xyxy_norm(cx, cy, bw, bh)
        x1 = (x1 * w0).clamp(0, w0)
        x2 = (x2 * w0).clamp(0, w0)
        y1 = (y1 * h0).clamp(0, h0)
        y2 = (y2 * h0).clamp(0, h0)
        x1, x2 = torch.minimum(x1, x2), torch.maximum(x1, x2)
        y1, y2 = torch.minimum(y1, y2), torch.maximum(y1, y2)

        order = max_score.argsort(descending=True)[: self.max_det]
        detections = []
        for i in order.tolist():
            cid = int(cls_ids[i])
            detections.append(
                Detection(
                    bbox=(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                    confidence=float(max_score[i]),
                    class_id=cid,
                    class_name=self.class_names[cid] if self.class_names and cid < len(self.class_names) else None,
                )
            )
        return detections
