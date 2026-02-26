"""
跟踪器共享工具函数测试（IoU 计算、线性分配）。

验证共享工具模块是否正常工作。
"""

import numpy as np
import pytest

from visionframework import (
    calculate_iou,
    iou_cost_matrix,
    linear_assignment,
    SCIPY_AVAILABLE,
)


def test_calculate_iou_identical_boxes():
    """完全相同的框应返回 IoU = 1.0。"""
    box = (10.0, 20.0, 50.0, 80.0)
    iou = calculate_iou(box, box)
    assert iou == 1.0


def test_calculate_iou_no_overlap():
    """完全不重叠的框应返回 IoU = 0.0。"""
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (20.0, 20.0, 30.0, 30.0)
    iou = calculate_iou(box1, box2)
    assert iou == 0.0


def test_calculate_iou_partial_overlap():
    """部分重叠的框应返回 0 < IoU < 1。"""
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (5.0, 5.0, 15.0, 15.0)
    iou = calculate_iou(box1, box2)
    assert 0.0 < iou < 1.0
    expected = 25.0 / 175.0
    assert abs(iou - expected) < 1e-6


def test_iou_cost_matrix_empty():
    """空列表应返回空矩阵。"""
    cost = iou_cost_matrix([], [])
    assert cost.shape == (0, 0)

    cost = iou_cost_matrix([(0, 0, 1, 1)], [])
    assert cost.shape == (1, 0)

    cost = iou_cost_matrix([], [(0, 0, 1, 1)])
    assert cost.shape == (0, 1)


def test_iou_cost_matrix_values():
    """验证 cost = 1 - IoU。"""
    boxes_a = [(0.0, 0.0, 10.0, 10.0)]
    boxes_b = [(0.0, 0.0, 10.0, 10.0), (20.0, 20.0, 30.0, 30.0)]

    cost = iou_cost_matrix(boxes_a, boxes_b)
    assert cost.shape == (1, 2)
    assert abs(cost[0, 0] - 0.0) < 1e-6
    assert abs(cost[0, 1] - 1.0) < 1e-6


def test_linear_assignment_empty():
    """空矩阵应返回空匹配。"""
    cost = np.zeros((0, 0))
    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, thresh=0.5)

    assert matches.shape == (0, 2)
    assert len(unmatched_rows) == 0
    assert len(unmatched_cols) == 0


def test_linear_assignment_perfect_match():
    """低成本矩阵应匹配所有行列。"""
    cost = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ])

    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, thresh=0.5)

    assert matches.shape[0] == 2
    assert len(unmatched_rows) == 0
    assert len(unmatched_cols) == 0


def test_linear_assignment_threshold_filter():
    """高于阈值的匹配应被过滤。"""
    cost = np.array([
        [0.2, 0.8],
        [0.9, 0.3],
    ])

    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, thresh=0.5)
    assert matches.shape[0] == 2


def test_linear_assignment_with_scipy_vs_greedy():
    """验证方阵的完全匹配结果。"""
    cost = np.array([
        [0.1, 0.9, 0.8],
        [0.7, 0.2, 0.6],
        [0.5, 0.4, 0.3],
    ])

    matches, unmatched_rows, unmatched_cols = linear_assignment(cost, thresh=1.0)

    assert matches.shape[0] == 3
    assert len(unmatched_rows) == 0
    assert len(unmatched_cols) == 0

    matched_rows = set(matches[:, 0])
    matched_cols = set(matches[:, 1])
    assert len(matched_rows) == 3
    assert len(matched_cols) == 3


def test_scipy_availability_flag():
    """SCIPY_AVAILABLE 应为布尔值。"""
    assert isinstance(SCIPY_AVAILABLE, bool)
