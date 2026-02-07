"""
Shared utilities for tracker implementations.

Consolidates IoU calculation, cost-matrix construction, and linear assignment
so that byte_tracker, iou_tracker, and reid_tracker don't each carry their own copy.
"""

import numpy as np
from typing import Tuple, List

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def calculate_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Calculate IoU (Intersection over Union) between two (x1, y1, x2, y2) boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def iou_cost_matrix(
    boxes_a: List[Tuple[float, float, float, float]],
    boxes_b: List[Tuple[float, float, float, float]],
) -> np.ndarray:
    """Build an (len(a), len(b)) cost matrix where cost = 1 - IoU."""
    m, n = len(boxes_a), len(boxes_b)
    if m == 0 or n == 0:
        return np.zeros((m, n), dtype=np.float32)

    cost = np.zeros((m, n), dtype=np.float32)
    for i, ba in enumerate(boxes_a):
        for j, bb in enumerate(boxes_b):
            cost[i, j] = 1.0 - calculate_iou(ba, bb)
    return cost


def linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run Hungarian (scipy) or greedy assignment, filtered by *thresh*.

    Returns:
        matches: (K, 2) array of (row, col) pairs
        unmatched_rows: 1-D array of unmatched row indices
        unmatched_cols: 1-D array of unmatched col indices
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0]),
            np.arange(cost_matrix.shape[1]),
        )

    if SCIPY_AVAILABLE:
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        matched = np.column_stack((row_idx, col_idx))
    else:
        # Greedy fallback
        matched_list: List[List[int]] = []
        used_r, used_c = set(), set()
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                if i not in used_r and j not in used_c and cost_matrix[i, j] < thresh:
                    matched_list.append([i, j])
                    used_r.add(i)
                    used_c.add(j)
        matched = np.array(matched_list, dtype=int) if matched_list else np.empty((0, 2), dtype=int)

    # Filter by threshold
    if matched.shape[0] > 0:
        keep = cost_matrix[matched[:, 0], matched[:, 1]] < thresh
        matched = matched[keep]

    matched_r = set(matched[:, 0]) if matched.shape[0] else set()
    matched_c = set(matched[:, 1]) if matched.shape[0] else set()
    unmatched_r = np.array([i for i in range(cost_matrix.shape[0]) if i not in matched_r], dtype=int)
    unmatched_c = np.array([j for j in range(cost_matrix.shape[1]) if j not in matched_c], dtype=int)

    return matched, unmatched_r, unmatched_c
