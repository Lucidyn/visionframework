"""
Tests for visualization, evaluation, error handling and dependency management utilities.

这些测试覆盖：
- Visualizer 基本绘制接口（检测 / 轨迹 / 统一 draw_results）
- DetectionEvaluator / TrackingEvaluator 基础指标计算
- ErrorHandler（handle_error / wrap_error / validate_input / format_error_message）
- DependencyManager 及其辅助函数的基本行为（在缺失依赖时安全返回）
"""

from typing import List, Dict, Any

import numpy as np
import pytest

from visionframework import Detection, Track, Pose, KeyPoint, Visualizer
from visionframework.exceptions import VisionFrameworkError
from visionframework.utils import (
    DetectionEvaluator,
    TrackingEvaluator,
    ErrorHandler,
    DependencyManager,
    dependency_manager,
    is_dependency_available,
    get_available_dependencies,
    get_missing_dependencies,
    validate_dependency,
    get_install_command,
    import_optional_dependency,
)


def _dummy_image(h: int = 240, w: int = 320) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_visualizer_draw_detections_and_results() -> None:
    """Visualizer 应该能在不报错的情况下绘制检测和轨迹结果，并保持图像形状。"""
    image = _dummy_image()

    detections = [
        Detection(bbox=(50, 60, 150, 180), confidence=0.9, class_id=0, class_name="person"),
    ]
    tracks = [
        Track(track_id=1, bbox=(55, 65, 145, 175), confidence=0.8, class_id=0, class_name="person"),
    ]

    # 构造一个简单的 Pose（只用 1 个关键点即可）
    keypoint = KeyPoint(x=100, y=120, confidence=0.95, keypoint_id=0, keypoint_name="head")
    pose = Pose(bbox=(80, 90, 120, 150), keypoints=[keypoint], confidence=0.95, pose_id=None)
    poses = [pose]

    viz = Visualizer()

    img_det = viz.draw_detections(image.copy(), detections)
    assert img_det.shape == image.shape

    img_tracks = viz.draw_tracks(image.copy(), tracks)
    assert img_tracks.shape == image.shape

    img_poses = viz.draw_poses(image.copy(), poses)
    assert img_poses.shape == image.shape

    img_all = viz.draw_results(image.copy(), detections=detections, tracks=tracks, poses=poses)
    assert img_all.shape == image.shape


def test_detection_evaluator_metrics() -> None:
    """DetectionEvaluator 在简单场景下应给出合理的 TP/FP/FN/precision/recall。"""
    evaluator = DetectionEvaluator(iou_threshold=0.5)

    gt = [
        Detection(bbox=(0, 0, 10, 10), confidence=1.0, class_id=0),
        Detection(bbox=(20, 20, 30, 30), confidence=1.0, class_id=0),
    ]
    preds = [
        Detection(bbox=(0, 0, 10, 10), confidence=0.9, class_id=0),  # perfect match
        Detection(bbox=(50, 50, 60, 60), confidence=0.8, class_id=0),  # FP
    ]

    metrics = evaluator.calculate_metrics(preds, gt)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0


def test_tracking_evaluator_mota_and_motp() -> None:
    """TrackingEvaluator 的 MOTA/MOTP 在简单场景下应可计算且数值合理。"""
    evaluator = TrackingEvaluator(iou_threshold=0.3)

    # 构造两帧简单的预测和 GT 轨迹
    pred_tracks: List[List[Dict[str, Any]]] = [
        [
            {"track_id": 1, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
        ],
        [
            {"track_id": 1, "bbox": {"x1": 2, "y1": 2, "x2": 12, "y2": 12}},
        ],
    ]
    gt_tracks: List[List[Dict[str, Any]]] = [
        [
            {"track_id": 100, "bbox": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
        ],
        [
            {"track_id": 100, "bbox": {"x1": 3, "y1": 3, "x2": 13, "y2": 13}},
        ],
    ]

    mota = evaluator.calculate_mota(pred_tracks, gt_tracks)
    motp = evaluator.calculate_motp(pred_tracks, gt_tracks)

    assert "MOTA" in mota and "precision" in mota and "recall" in mota
    assert mota["total_gt"] > 0
    assert motp["total_matched_pairs"] >= 0


def test_error_handler_basic_usage() -> None:
    """ErrorHandler 的基本方法应能正常工作。"""
    handler = ErrorHandler()

    # 1. handle_error 返回一个异常对象
    class DummyError(VisionFrameworkError):
        pass

    original = ValueError("boom")
    exc = handler.handle_error(
        error=original,
        error_type=DummyError,
        message="测试错误",
        context={"step": "unit_test"},
        raise_error=False,
        log_traceback=False,
    )
    assert exc is not None

    # 2. wrap_error 在出错时返回默认值
    def bad_func():
        raise RuntimeError("fail")

    wrapped = handler.wrap_error(
        func=bad_func,
        error_type=DummyError,
        message="包装错误",
        default_return="fallback",
        raise_error=False,
    )
    assert wrapped() == "fallback"

    # 3. validate_input
    ok, msg = handler.validate_input({"a": 1}, expected_type=dict, param_name="x")
    assert ok and msg is None
    ok, msg = handler.validate_input("bad", expected_type=dict, param_name="x")
    assert not ok and isinstance(msg, str)

    # 4. format_error_message
    msg = handler.format_error_message("测试", error=original, context={"a": 1})
    assert "测试" in msg and "Original error" in msg


def test_dependency_manager_and_helpers() -> None:
    """
    DependencyManager 及助手函数在缺少依赖时应安全返回：
    - is_available / is_dependency_available 为 False
    - import_optional_dependency 返回 None
    - get_install_command 至少能返回字符串（对已定义依赖）
    """
    manager = DependencyManager()

    # 对已定义依赖（如 clip/sam 等），在当前环境可能不可用，但调用必须安全
    for dep in manager.OPTIONAL_DEPENDENCIES.keys():
        available = is_dependency_available(dep)
        status = manager.get_dependency_status(dep)
        assert "available" in status and "message" in status
        cmd = get_install_command(dep)
        if cmd is not None:
            assert isinstance(cmd, str)

    # import_optional_dependency 在缺失依赖时应返回 None
    mod = import_optional_dependency("nonexistent_dep", "nonexistent_pkg")
    assert mod is None

    # 全局函数 get_available_dependencies / get_missing_dependencies 只需能被安全调用
    _ = get_available_dependencies()
    _ = get_missing_dependencies()

