"""
Basic tests for high-level detection / tracking APIs.

这些测试不依赖真实模型权重，只验证：
- create_detector / create_pipeline 是否能返回对象
- process_image / VisionPipeline.process 在模型初始化失败时是否稳定返回结构化结果
"""

from typing import Dict, Any

import numpy as np

from visionframework import (
    YOLODetector,
    VisionPipeline,
    create_detector,
    create_pipeline,
    process_image,
)


def _make_dummy_image(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_create_detector_returns_yolo_detector_even_if_init_fails() -> None:
    """
    create_detector 在权重不存在时不会抛异常，而是返回一个检测器实例。
    初始化失败会在内部通过日志处理。
    """
    det = create_detector(
        model_path="nonexistent_model.pt",  # 故意不存在的路径
        model_type="yolo",
        device="cpu",
        conf_threshold=0.5,
    )
    assert isinstance(det, YOLODetector)


def test_process_image_returns_expected_keys() -> None:
    """
    process_image 在模型加载失败时，仍应返回包含 detections / tracks / poses 的字典。
    """
    img = _make_dummy_image()

    result: Dict[str, Any] = process_image(
        img,
        model_path="nonexistent_model.pt",  # 故意让初始化失败
        enable_tracking=True,
        enable_segmentation=False,
        enable_pose_estimation=False,
    )

    assert isinstance(result, dict)
    assert "detections" in result
    assert "tracks" in result
    assert "poses" in result
    assert isinstance(result["detections"], list)
    assert isinstance(result["tracks"], list)
    assert isinstance(result["poses"], list)


def test_vision_pipeline_with_tracking_process_structure() -> None:
    """
    直接使用 VisionPipeline，在 detector 初始化失败时，process 也应返回结构化结果。
    """
    config = {
        "detector_config": {
            "model_path": "nonexistent_model.pt",  # 故意不存在
            "device": "cpu",
        },
        "enable_tracking": True,
        "tracker_config": {
            "tracker_type": "bytetrack",
        },
    }
    pipeline = VisionPipeline(config)
    img = _make_dummy_image()

    result = pipeline.process(img)
    assert isinstance(result, dict)
    assert "detections" in result
    assert "tracks" in result
    assert "poses" in result


def test_create_pipeline_and_process_image() -> None:
    """
    使用 create_pipeline 创建管道并处理图像，验证高层封装是否可用。
    """
    pipeline = create_pipeline(
        detector_config={"model_path": "nonexistent_model.pt", "device": "cpu"},
        enable_tracking=False,
        enable_segmentation=False,
        enable_pose_estimation=False,
    )
    img = _make_dummy_image()
    result = pipeline.process(img)

    assert isinstance(result, dict)
    assert "detections" in result
    assert isinstance(result["detections"], list)

