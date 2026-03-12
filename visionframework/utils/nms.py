"""
Non-Maximum Suppression (NMS).
"""

import numpy as np
from typing import List, Optional


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
    class_ids: Optional[np.ndarray] = None,
) -> List[int]:
    """Greedy NMS on (x1, y1, x2, y2) boxes.

    When ``class_ids`` is given, NMS is performed per-class by offsetting
    box coordinates so that boxes of different classes never overlap.

    Returns indices of kept boxes sorted by descending score.
    """
    if len(boxes) == 0:
        return []

    work_boxes = boxes.copy()
    if class_ids is not None:
        offsets = class_ids.astype(np.float64) * 4096.0
        work_boxes[:, 0] += offsets
        work_boxes[:, 1] += offsets
        work_boxes[:, 2] += offsets
        work_boxes[:, 3] += offsets

    x1, y1, x2, y2 = work_boxes[:, 0], work_boxes[:, 1], work_boxes[:, 2], work_boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]

    return keep
