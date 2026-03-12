"""
Detection algorithm: wraps a detection model with preprocessing and NMS.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torchF
from typing import Any, Dict, List, Optional, Tuple

from visionframework.core.registry import ALGORITHMS
from visionframework.data.detection import Detection
from visionframework.utils.bbox import xywh2xyxy
from visionframework.utils.nms import non_max_suppression
from visionframework.utils.device import resolve_device


@ALGORITHMS.register("Detector")
class Detector:
    """Detection algorithm.

    Wraps a ``ModelWrapper`` (backbone → neck → head) and adds:
    * letterbox preprocessing
    * DFL box decoding
    * confidence filtering + NMS

    Parameters
    ----------
    model : nn.Module
        Built detection model (via ``build_model``).
    input_size : tuple[int, int]
        ``(height, width)`` the network expects.
    conf : float
        Confidence threshold.
    nms_iou : float
        IoU threshold for NMS.
    class_names : list[str] | None
        Optional list mapping class-id → name.
    device : str
        ``'cpu'``, ``'cuda'``, or ``'auto'``.
    fp16 : bool
        Use half-precision inference on CUDA.
    """

    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, int] = (640, 640),
        conf: float = 0.25,
        nms_iou: float = 0.45,
        class_names: Optional[List[str]] = None,
        filter_classes: Optional[List] = None,
        device: str = "auto",
        fp16: bool = False,
        **_kw,
    ):
        self.input_size = input_size
        self.conf = conf
        self.nms_iou = nms_iou
        self.class_names = class_names
        self.fp16 = fp16 and torch.cuda.is_available()

        self._filter_ids = self._resolve_filter(filter_classes, class_names)

        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        if self.fp16:
            self.model = self.model.half()

    @staticmethod
    def _resolve_filter(filter_classes, class_names):
        """将 filter_classes（int/str 混合列表）统一解析为 class_id 集合。"""
        if not filter_classes:
            return None
        ids = set()
        name_to_id = {}
        if class_names:
            name_to_id = {n.lower(): i for i, n in enumerate(class_names) if n}
        for c in filter_classes:
            if isinstance(c, int):
                ids.add(c)
            elif isinstance(c, str):
                cid = name_to_id.get(c.lower())
                if cid is not None:
                    ids.add(cid)
        return ids if ids else None

    # -- preprocessing -------------------------------------------------------

    def _preprocess(self, img: np.ndarray) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
        """Letterbox resize → normalise → tensor."""
        h0, w0 = img.shape[:2]
        th, tw = self.input_size
        scale = min(th / h0, tw / w0)
        nh, nw = int(h0 * scale), int(w0 * scale)
        resized = cv2.resize(img, (nw, nh))

        pad_h, pad_w = th - nh, tw - nw
        top, left = pad_h // 2, pad_w // 2
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        canvas[top:top + nh, left:left + nw] = resized

        tensor = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        if self.fp16:
            tensor = tensor.half()
        return tensor, scale, (top, left)

    # -- postprocessing ------------------------------------------------------

    def _decode_outputs(self, raw_outputs, scale, pad):
        """Decode raw head outputs into ``(xyxy, conf, cls_id)`` numpy arrays."""
        all_boxes, all_scores = [], []
        top, left = pad

        for cls_logits, reg_raw in raw_outputs:
            B, C, H, W = cls_logits.shape
            scores = cls_logits.sigmoid()  # (B, nc, H, W)

            reg_ch = reg_raw.shape[1]
            reg_max = reg_ch // 4
            if reg_max > 1:
                reg = reg_raw.view(B, 4, reg_max, H, W).softmax(dim=2)
                proj = torch.arange(reg_max, device=reg.device, dtype=reg.dtype)
                proj = proj.view(1, 1, -1, 1, 1)
                reg = (reg * proj).sum(dim=2)  # (B, 4, H, W) — DFL decode
            else:
                reg = reg_raw  # (B, 4, H, W) — direct ltrb

            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=scores.device, dtype=scores.dtype),
                torch.arange(W, device=scores.device, dtype=scores.dtype),
                indexing="ij",
            )
            stride_h = self.input_size[0] / H
            stride_w = self.input_size[1] / W

            x_c = (grid_x + 0.5) * stride_w
            y_c = (grid_y + 0.5) * stride_h
            x1 = x_c - reg[:, 0] * stride_w
            y1 = y_c - reg[:, 1] * stride_h
            x2 = x_c + reg[:, 2] * stride_w
            y2 = y_c + reg[:, 3] * stride_h

            boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(B, -1, 4)
            scores_flat = scores.permute(0, 2, 3, 1).reshape(B, -1, C)

            all_boxes.append(boxes)
            all_scores.append(scores_flat)

        boxes = torch.cat(all_boxes, dim=1)
        scores = torch.cat(all_scores, dim=1)

        boxes[..., [0, 2]] = (boxes[..., [0, 2]] - left) / scale
        boxes[..., [1, 3]] = (boxes[..., [1, 3]] - top) / scale

        return boxes[0], scores[0]

    # -- public API ----------------------------------------------------------

    @torch.no_grad()
    def predict(self, img: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR image and return ``Detection`` objects."""
        tensor, scale, pad = self._preprocess(img)
        raw = self.model(tensor)
        boxes, scores = self._decode_outputs(raw, scale, pad)

        max_scores, cls_ids = scores.max(dim=1)
        keep_mask = max_scores > self.conf
        boxes = boxes[keep_mask]
        max_scores = max_scores[keep_mask]
        cls_ids = cls_ids[keep_mask]

        if boxes.numel() == 0:
            return []

        if self._filter_ids is not None:
            cls_mask = torch.tensor(
                [int(c) in self._filter_ids for c in cls_ids],
                dtype=torch.bool, device=boxes.device,
            )
            boxes = boxes[cls_mask]
            max_scores = max_scores[cls_mask]
            cls_ids = cls_ids[cls_mask]

        if boxes.numel() == 0:
            return []

        nms_keep = non_max_suppression(
            boxes.cpu().numpy(),
            max_scores.cpu().numpy(),
            self.nms_iou,
            class_ids=cls_ids.cpu().numpy(),
        )
        detections = []
        for idx in nms_keep[:300]:
            b = boxes[idx].cpu().numpy()
            detections.append(Detection(
                bbox=tuple(b.tolist()),
                confidence=float(max_scores[idx]),
                class_id=int(cls_ids[idx]),
                class_name=(self.class_names[int(cls_ids[idx])]
                            if self.class_names else None),
            ))
        return detections

    @torch.no_grad()
    def predict_batch(self, images: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on a batch of images."""
        return [self.predict(img) for img in images]
