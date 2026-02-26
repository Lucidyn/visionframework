"""
模型工具与分析工具测试。

覆盖：轻量级 CPU 代码路径和基本 API 契约。
"""

import math
from typing import List

import numpy as np
import pytest
import torch

from visionframework import (
    Track,
    TrajectoryAnalyzer,
    AugmentationConfig,
    AugmentationType,
    ImageAugmenter,
    DistillationConfig,
    PruningConfig,
    prune_model,
    QuantizationConfig,
    quantize_model,
    compare_model_performance,
)


class _TinyNet(torch.nn.Module):
    """用于快速测试的极小网络。"""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 4)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


@pytest.mark.parametrize("backend", ["fbgemm", "qnnpack"])
def test_quantization_dynamic_cpu(backend: str) -> None:
    """动态量化应成功，或在后端不可用时跳过。"""
    model = _TinyNet().eval()

    config = QuantizationConfig(
        quantization_type="dynamic",
        backend=backend,
        verbose=False,
    )

    try:
        q_model = quantize_model(model, config)
    except (RuntimeError, ValueError) as e:
        pytest.skip(f"量化后端 '{backend}' 不可用：{e}")

    assert isinstance(q_model, torch.nn.Module)
    x = torch.randn(1, 8)
    with torch.no_grad():
        out = q_model(x)
    assert out.shape == (1, 2)


def test_pruning_basic() -> None:
    """剪枝应运行并保持模型结构完整。"""
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

    x = torch.randn(2, 8)
    out = pruned(x)
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()


def test_compare_model_performance_dummy() -> None:
    """compare_model_performance 应返回包含大小和时间信息的字典。"""
    model = _TinyNet().eval()
    optimized = _TinyNet().eval()

    test_data = [torch.randn(1, 8) for _ in range(3)]
    metrics = compare_model_performance(model, optimized, test_data)

    assert metrics["original_size"] > 0
    assert metrics["optimized_size"] > 0
    assert "size_reduction" in metrics
    assert metrics["original_inference_time"] >= 0.0
    assert metrics["optimized_inference_time"] >= 0.0
    assert "speedup" in metrics


def test_distillation_config_defaults() -> None:
    """DistillationConfig 应提供合理的默认值。"""
    cfg = DistillationConfig()
    assert cfg.temperature > 0
    assert 0 < cfg.alpha <= 1
    assert cfg.student_loss_weight >= 0
    assert cfg.epochs > 0
    assert cfg.batch_size > 0


def _make_dummy_image(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def test_image_augmenter_single_and_batch() -> None:
    """ImageAugmenter 对单张和批量图像应保持形状。"""
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
    assert aug.ndim == 3
    assert aug.shape[2] == 3
    assert aug.shape[0] > 0 and aug.shape[1] > 0
    assert aug.dtype == img.dtype

    batch: List[np.ndarray] = [_make_dummy_image() for _ in range(3)]
    aug_batch = [augmenter.augment(im) for im in batch]
    assert len(aug_batch) == len(batch)
    for a, b in zip(aug_batch, batch):
        assert a.ndim == 3
        assert a.shape[2] == b.shape[2] == 3
        assert a.shape[0] > 0 and a.shape[1] > 0
        assert a.dtype == b.dtype


def test_trajectory_analyzer_speed_and_direction() -> None:
    """TrajectoryAnalyzer 基础速度和方向计算。"""
    analyzer = TrajectoryAnalyzer(fps=30.0, pixel_to_meter=0.1)

    track = Track(track_id=1, bbox=(0, 0, 10, 10), confidence=1.0, class_id=0)
    sx, sy = analyzer.calculate_speed(track, use_real_world=True)
    assert sx == pytest.approx(0.0)
    assert sy == pytest.approx(0.0)

    track.update((10, 0, 20, 10), confidence=1.0)
    sx, sy = analyzer.calculate_speed(track, use_real_world=False)
    assert sx != 0.0 or sy != 0.0

    direction = analyzer.calculate_direction(track)
    assert abs(direction) < 45.0
