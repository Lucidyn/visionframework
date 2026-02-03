"""
Tests for new model tools and analysis utilities.
These tests focus on light, CPU-only code paths and basic API contracts.
"""

import math
from typing import List

import numpy as np
import pytest
import torch

from visionframework.data import Track
from visionframework.utils import TrajectoryAnalyzer
from visionframework.utils.data_augmentation import (
    AugmentationConfig,
    AugmentationType,
    ImageAugmenter,
)
from visionframework.utils.model_optimization import (
    DistillationConfig,
    PruningConfig,
    prune_model,
    QuantizationConfig,
    quantize_model,
    compare_model_performance,
)


class _TinyNet(torch.nn.Module):
    """Very small network for fast tests."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 4)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        return self.fc2(self.relu(self.fc1(x)))


@pytest.mark.parametrize("backend", ["fbgemm", "qnnpack"])
def test_quantization_dynamic_cpu(backend: str) -> None:
    """Dynamic quantization should succeed or be cleanly skipped when backend is unavailable."""
    model = _TinyNet().eval()

    config = QuantizationConfig(
        quantization_type="dynamic",
        backend=backend,
        verbose=False,
    )

    try:
        q_model = quantize_model(model, config)
    except (RuntimeError, ValueError) as e:
        # Some backends are not available on all platforms (e.g. Windows CPU).
        pytest.skip(f"Quantization backend '{backend}' not available: {e}")

    assert isinstance(q_model, torch.nn.Module)
    # Forward should still work on simple input
    x = torch.randn(1, 8)
    with torch.no_grad():
        out = q_model(x)
    assert out.shape == (1, 2)


def test_pruning_basic() -> None:
    """Pruning should run and keep the model structure intact."""
    model = _TinyNet()

    config = PruningConfig(
        pruning_type="l1_unstructured",
        amount=0.2,
        target_modules=[torch.nn.Linear],
        global_pruning=False,
        verbose=False,
    )

    pruned = prune_model(model, config)
    assert isinstance(pruned, _TinyNet)

    # Ensure parameters still produce finite outputs
    x = torch.randn(2, 8)
    out = pruned(x)
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


def test_compare_model_performance_dummy() -> None:
    """compare_model_performance 应该返回包含大小和时间信息的字典。"""
    model = _TinyNet().eval()
    # 使用一个简单的“优化后模型”：拷贝一份即可
    optimized = _TinyNet().eval()

    test_data = [torch.randn(1, 8) for _ in range(3)]
    metrics = compare_model_performance(model, optimized, test_data)

    # 仅检查关键字段存在且为非负数，speedup 可能在数值上接近 1
    assert metrics["original_size"] > 0
    assert metrics["optimized_size"] > 0
    assert "size_reduction" in metrics
    assert metrics["original_inference_time"] >= 0.0
    assert metrics["optimized_inference_time"] >= 0.0
    # 当两者时间都非常小（接近 0）时，compare_model_performance 内部会避免除零
    assert "speedup" in metrics


def test_distillation_config_defaults() -> None:
    """DistillationConfig should provide sensible defaults."""
    cfg = DistillationConfig()
    assert cfg.temperature > 0
    assert 0 < cfg.alpha <= 1
    assert cfg.student_loss_weight >= 0
    assert cfg.epochs > 0
    assert cfg.batch_size > 0


def _make_dummy_image(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_image_augmenter_single_and_batch() -> None:
    """ImageAugmenter should preserve shape for single and batch augment."""
    config = AugmentationConfig(
        augmentations=[
            AugmentationType.FLIP,
            AugmentationType.ROTATE,
            AugmentationType.BRIGHTNESS,
        ],
        probability=0.8,
        random_order=True,
        seed=123,
    )
    augmenter = ImageAugmenter(config)

    img = _make_dummy_image()
    aug = augmenter.augment(img)
    # 增强后图像形状可能变化（如旋转、缩放），只检查基本属性
    assert aug.ndim == 3
    assert aug.shape[2] == 3
    assert aug.shape[0] > 0 and aug.shape[1] > 0
    assert aug.dtype == img.dtype

    batch: List[np.ndarray] = [_make_dummy_image() for _ in range(3)]
    # 批量增强：逐张调用 augment
    aug_batch = [augmenter.augment(im) for im in batch]
    assert len(aug_batch) == len(batch)
    for a, b in zip(aug_batch, batch):
        assert a.ndim == 3
        assert a.shape[2] == b.shape[2] == 3
        assert a.shape[0] > 0 and a.shape[1] > 0
        assert a.dtype == b.dtype


def test_trajectory_analyzer_speed_and_direction() -> None:
    """TrajectoryAnalyzer basic speed and direction calculations."""
    analyzer = TrajectoryAnalyzer(fps=30.0, pixel_to_meter=0.1)

    # Track with only one history point -> zero speed
    track = Track(track_id=1, bbox=(0, 0, 10, 10), confidence=1.0, class_id=0)
    sx, sy = analyzer.calculate_speed(track, use_real_world=True)
    assert sx == pytest.approx(0.0)
    assert sy == pytest.approx(0.0)

    # Add another point with some movement
    track.update((10, 0, 20, 10), confidence=1.0)
    sx, sy = analyzer.calculate_speed(track, use_real_world=False)
    assert sx != 0.0 or sy != 0.0

    direction = analyzer.calculate_direction(track)
    # Movement is mostly along +x axis, so direction should be near 0 degrees
    assert abs(direction) < 45.0


