"""
Tests for the Vision API.

验证 Vision 类的两种创建方式以及 run() 方法。
不依赖真实模型权重。
"""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest

from visionframework import Vision


def _make_dummy_image(h: int = 480, w: int = 640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ================================================================
# 1. Vision() — keyword constructor
# ================================================================

def test_vision_constructor_defaults() -> None:
    """Vision() 使用默认参数应能创建实例（即使模型不存在）。"""
    v = Vision(model="nonexistent_model.pt")
    assert isinstance(v, Vision)
    assert "nonexistent_model.pt" in repr(v)


def test_vision_constructor_with_tracking() -> None:
    """Vision(track=True) 应能创建带跟踪功能的实例。"""
    v = Vision(model="nonexistent_model.pt", track=True)
    assert "track=" in repr(v)


def test_vision_run_single_image() -> None:
    """Vision.run() 对单张图片应返回迭代器，每次迭代返回 (frame, meta, result)。"""
    v = Vision(model="nonexistent_model.pt")
    img = _make_dummy_image()

    for frame, meta, result in v.run(img):
        assert isinstance(frame, np.ndarray)
        assert isinstance(meta, dict)
        assert isinstance(result, dict)
        assert "detections" in result
        assert "tracks" in result
        assert "poses" in result
        assert isinstance(result["detections"], list)
        assert isinstance(result["tracks"], list)
        assert isinstance(result["poses"], list)


def test_vision_run_with_tracking() -> None:
    """Vision(track=True).run() 应该返回包含 tracks 键的结果。"""
    v = Vision(model="nonexistent_model.pt", track=True)
    img = _make_dummy_image()

    for frame, meta, result in v.run(img):
        assert "tracks" in result


# ================================================================
# 2. Vision.from_config() — config-based constructor
# ================================================================

def test_vision_from_config_dict() -> None:
    """Vision.from_config(dict) 应能从字典创建实例。"""
    cfg = {
        "model": "nonexistent_model.pt",
        "track": True,
        "conf": 0.3,
    }
    v = Vision.from_config(cfg)
    assert isinstance(v, Vision)
    assert "track=" in repr(v)


def test_vision_from_config_json_file() -> None:
    """Vision.from_config('file.json') 应能从 JSON 文件创建实例。"""
    cfg = {
        "model": "nonexistent_model.pt",
        "model_type": "yolo",
        "device": "cpu",
        "conf": 0.25,
        "track": False,
    }

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(cfg, f)
        f.flush()
        config_path = f.name

    try:
        v = Vision.from_config(config_path)
        assert isinstance(v, Vision)
    finally:
        Path(config_path).unlink(missing_ok=True)


def test_vision_from_config_missing_file() -> None:
    """Vision.from_config() 对不存在的文件应抛出 FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        Vision.from_config("nonexistent_config.json")


def test_vision_from_config_unsupported_format() -> None:
    """Vision.from_config() 对不支持的文件格式应抛出 ValueError。"""
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        f.write(b"<config/>")
        f.flush()
        path = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported"):
            Vision.from_config(path)
    finally:
        Path(path).unlink(missing_ok=True)


# ================================================================
# 3. run() — result structure
# ================================================================

def test_vision_run_result_structure() -> None:
    """验证 run() 返回的 result 结构完整。"""
    v = Vision(model="nonexistent_model.pt", track=True, pose=True)
    img = _make_dummy_image()

    for frame, meta, result in v.run(img):
        # frame 应该是原始图片
        assert frame.shape == img.shape
        # result 应该有三个键
        assert set(result.keys()) >= {"detections", "tracks", "poses"}


# ================================================================
# 4. cleanup
# ================================================================

def test_vision_cleanup() -> None:
    """cleanup() 应该不抛出异常。"""
    v = Vision(model="nonexistent_model.pt")
    v.cleanup()  # 不应该抛异常


# ================================================================
# 5. repr
# ================================================================

def test_vision_repr() -> None:
    """repr(v) 应该是人类可读的。"""
    v = Vision(model="yolov8n.pt", track=True, pose=True, device="cpu")
    r = repr(v)
    assert "Vision(" in r
    assert "yolov8n.pt" in r
    assert "track=" in r
    assert "pose=True" in r


def test_vision_repr_fp16() -> None:
    """repr 应该显示 fp16 标志。"""
    v = Vision(model="nonexistent_model.pt", fp16=True)
    assert "fp16=True" in repr(v)


def test_vision_repr_batch() -> None:
    """repr 应该显示 batch_inference 标志。"""
    v = Vision(model="nonexistent_model.pt", batch_inference=True)
    assert "batch_inference=True" in repr(v)


# ================================================================
# 6. 新增参数 — fp16, batch, category_thresholds
# ================================================================

def test_vision_constructor_fp16() -> None:
    """Vision(fp16=True) 应能创建实例。"""
    v = Vision(model="nonexistent_model.pt", fp16=True, device="cpu")
    assert isinstance(v, Vision)


def test_vision_constructor_batch_inference() -> None:
    """Vision(batch_inference=True) 应能创建实例。"""
    v = Vision(model="nonexistent_model.pt", batch_inference=True)
    assert isinstance(v, Vision)


def test_vision_constructor_dynamic_batch() -> None:
    """Vision(dynamic_batch=True) 应能创建实例。"""
    v = Vision(model="nonexistent_model.pt", dynamic_batch=True, max_batch_size=16)
    assert isinstance(v, Vision)


def test_vision_constructor_category_thresholds() -> None:
    """Vision(category_thresholds={...}) 应能创建实例。"""
    v = Vision(
        model="nonexistent_model.pt",
        category_thresholds={"person": 0.6, "car": 0.3},
    )
    assert isinstance(v, Vision)


def test_vision_from_config_with_new_params() -> None:
    """Vision.from_config() 应能解析 fp16 / batch / category_thresholds。"""
    cfg = {
        "model": "nonexistent_model.pt",
        "fp16": True,
        "batch_inference": True,
        "dynamic_batch": True,
        "max_batch_size": 16,
        "category_thresholds": {"person": 0.5},
    }
    v = Vision.from_config(cfg)
    assert isinstance(v, Vision)
    assert "fp16=True" in repr(v)
    assert "batch_inference=True" in repr(v)


def test_vision_run_with_fp16() -> None:
    """fp16 模式下 run() 应正常返回结果。"""
    v = Vision(model="nonexistent_model.pt", fp16=True)
    img = _make_dummy_image()
    for frame, meta, result in v.run(img):
        assert isinstance(result, dict)
        assert "detections" in result


def test_vision_all_params_combined() -> None:
    """所有参数组合使用应能创建实例并运行。"""
    v = Vision(
        model="nonexistent_model.pt",
        model_type="yolo",
        device="cpu",
        conf=0.3,
        iou=0.5,
        track=True,
        tracker="bytetrack",
        segment=True,
        pose=True,
        fp16=False,
        batch_inference=True,
        dynamic_batch=True,
        max_batch_size=4,
        category_thresholds={"person": 0.6},
    )
    assert isinstance(v, Vision)
    img = _make_dummy_image()
    for frame, meta, result in v.run(img):
        assert isinstance(result, dict)
