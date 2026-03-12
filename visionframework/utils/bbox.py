"""
Bounding-box format conversions and helpers.
"""

import numpy as np


def xyxy2xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert (x1, y1, x2, y2) to (cx, cy, w, h)."""
    out = np.empty_like(boxes)
    out[..., 0] = (boxes[..., 0] + boxes[..., 2]) / 2
    out[..., 1] = (boxes[..., 1] + boxes[..., 3]) / 2
    out[..., 2] = boxes[..., 2] - boxes[..., 0]
    out[..., 3] = boxes[..., 3] - boxes[..., 1]
    return out


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (cx, cy, w, h) to (x1, y1, x2, y2)."""
    out = np.empty_like(boxes)
    half_w = boxes[..., 2] / 2
    half_h = boxes[..., 3] / 2
    out[..., 0] = boxes[..., 0] - half_w
    out[..., 1] = boxes[..., 1] - half_h
    out[..., 2] = boxes[..., 0] + half_w
    out[..., 3] = boxes[..., 1] + half_h
    return out


def clip_boxes(boxes: np.ndarray, img_shape: tuple) -> np.ndarray:
    """Clip boxes to image boundaries (h, w)."""
    h, w = img_shape[:2]
    out = boxes.copy()
    out[..., [0, 2]] = out[..., [0, 2]].clip(0, w)
    out[..., [1, 3]] = out[..., [1, 3]].clip(0, h)
    return out
