"""
YOLO11 / YOLO26 实例分割（Ultralytics Segment 模型）。

依赖 ``ultralytics``；权重为官方 ``*-seg.pt``（或本地路径）。输出为带 ``mask`` 的
:class:`~visionframework.data.detection.Detection` 列表。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from visionframework.algorithms.base import BaseAlgorithm
from visionframework.core.registry import ALGORITHMS
from visionframework.data.detection import Detection
from visionframework.utils.device import resolve_device

logger = logging.getLogger(__name__)

_COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


class _UltralyticsSegmenterBase(BaseAlgorithm):
    """Ultralytics YOLO Segment 推理封装。"""

    _family: str = "yolo11"

    def __init__(
        self,
        weights: str,
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        fp16: bool = False,
        **_kw,
    ):
        super().__init__(model=None, device=device, fp16=fp16)
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "YOLO 实例分割需要安装 ultralytics：pip install ultralytics"
            ) from e

        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        wpath = str(weights)
        stem = Path(wpath).stem.lower()
        if self._family == "yolo11" and not stem.startswith("yolo11"):
            logger.warning(
                "YOLO11Segmenter 建议使用 yolo11*-seg 权重，当前: %s", wpath
            )
        if self._family == "yolo26" and not stem.startswith("yolo26"):
            logger.warning(
                "YOLO26Segmenter 建议使用 yolo26*-seg 权重，当前: %s", wpath
            )

        self._yolo = YOLO(wpath)
        self.device = resolve_device(device)

    def predict(self, img: np.ndarray) -> List[Detection]:
        """对 BGR 图像推理，返回带实例 mask 的检测列表。"""
        results = self._yolo.predict(
            source=img,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        h0, w0 = img.shape[:2]
        boxes = r.boxes
        n = len(boxes)
        detections: List[Detection] = []

        masks_data = None
        if r.masks is not None and r.masks.data is not None:
            masks_data = r.masks.data

        for i in range(n):
            xyxy = boxes.xyxy[i].detach().cpu().numpy()
            conf_v = float(boxes.conf[i].detach().cpu().item())
            cid = int(boxes.cls[i].detach().cpu().item())
            name = _COCO_NAMES[cid] if cid < len(_COCO_NAMES) else str(cid)

            mask: Optional[np.ndarray] = None
            if masks_data is not None and i < masks_data.shape[0]:
                m = masks_data[i].detach().cpu().numpy().astype(np.float32)
                if m.shape[0] != h0 or m.shape[1] != w0:
                    m = cv2.resize(m, (w0, h0), interpolation=cv2.INTER_LINEAR)
                mask = (m > 0.5).astype(np.uint8) * 255

            detections.append(
                Detection(
                    bbox=tuple(float(x) for x in xyxy.tolist()),
                    confidence=conf_v,
                    class_id=cid,
                    class_name=name,
                    mask=mask,
                )
            )
        return detections


@ALGORITHMS.register("YOLO11Segmenter")
class YOLO11Segmenter(_UltralyticsSegmenterBase):
    """YOLO11 Segment（``yolo11n-seg.pt`` … ``yolo11x-seg.pt``）。"""

    _family = "yolo11"


@ALGORITHMS.register("YOLO26Segmenter")
class YOLO26Segmenter(_UltralyticsSegmenterBase):
    """YOLO26 Segment（``yolo26n-seg.pt`` … ``yolo26x-seg.pt``）。"""

    _family = "yolo26"
