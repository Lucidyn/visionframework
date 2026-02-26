"""
ImageAugmenter 与 AugmentationConfig 测试。
"""

import numpy as np
import pytest

from visionframework import (
    ImageAugmenter,
    AugmentationConfig,
    AugmentationType,
    InterpolationType,
)


def _make_image(h: int = 128, w: int = 128) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# AugmentationConfig
# ---------------------------------------------------------------------------

def test_augmentation_config_defaults():
    cfg = AugmentationConfig(augmentations=[AugmentationType.FLIP])
    assert cfg.probability == 0.5
    assert cfg.random_order is True
    assert cfg.interpolation == InterpolationType.BILINEAR


def test_augmentation_config_custom():
    cfg = AugmentationConfig(
        augmentations=[AugmentationType.ROTATE, AugmentationType.BLUR],
        probability=0.8,
        random_order=False,
        seed=42,
    )
    assert cfg.probability == 0.8
    assert cfg.seed == 42


# ---------------------------------------------------------------------------
# ImageAugmenter — 创建
# ---------------------------------------------------------------------------

def test_augmenter_creation():
    cfg = AugmentationConfig(augmentations=[AugmentationType.FLIP])
    aug = ImageAugmenter(cfg)
    assert isinstance(aug, ImageAugmenter)


def test_augmenter_minimal_config():
    cfg = AugmentationConfig(augmentations=[AugmentationType.FLIP])
    aug = ImageAugmenter(cfg)
    assert isinstance(aug, ImageAugmenter)


# ---------------------------------------------------------------------------
# 各增强类型
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("aug_type", [
    AugmentationType.FLIP,
    AugmentationType.ROTATE,
    AugmentationType.SCALE,
    AugmentationType.BRIGHTNESS,
    AugmentationType.CONTRAST,
    AugmentationType.BLUR,
    AugmentationType.NOISE,
    AugmentationType.CUTOUT,
    AugmentationType.COLOR_JITTER,
])
def test_augment_single_type(aug_type):
    cfg = AugmentationConfig(augmentations=[aug_type], probability=1.0)
    aug = ImageAugmenter(cfg)
    img = _make_image()
    result = aug.augment(img)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.dtype == np.uint8


def test_augment_flip():
    cfg = AugmentationConfig(augmentations=[AugmentationType.FLIP], probability=1.0)
    aug = ImageAugmenter(cfg)
    img = _make_image()
    result = aug.augment(img)
    assert result.shape == img.shape


def test_augment_rotate():
    cfg = AugmentationConfig(augmentations=[AugmentationType.ROTATE], probability=1.0)
    aug = ImageAugmenter(cfg)
    img = _make_image()
    result = aug.augment(img)
    assert result.ndim == 3


def test_augment_output_shape_preserved():
    """大多数增强应保持空间维度。"""
    cfg = AugmentationConfig(
        augmentations=[AugmentationType.BRIGHTNESS, AugmentationType.CONTRAST],
        probability=1.0,
    )
    aug = ImageAugmenter(cfg)
    img = _make_image(64, 64)
    result = aug.augment(img)
    assert result.shape == img.shape


# ---------------------------------------------------------------------------
# 批量增强
# ---------------------------------------------------------------------------

def test_augment_multiple_images():
    cfg = AugmentationConfig(augmentations=[AugmentationType.FLIP], probability=1.0)
    aug = ImageAugmenter(cfg)
    images = [_make_image() for _ in range(4)]
    results = [aug.augment(img) for img in images]
    assert len(results) == 4
    for r in results:
        assert isinstance(r, np.ndarray)


# ---------------------------------------------------------------------------
# seed 参数
# ---------------------------------------------------------------------------

def test_augment_seed_config_accepted():
    """AugmentationConfig 接受 seed 参数不报错。"""
    cfg = AugmentationConfig(
        augmentations=[AugmentationType.BRIGHTNESS],
        probability=1.0,
        random_order=False,
        seed=123,
    )
    aug = ImageAugmenter(cfg)
    img = _make_image(64, 64)
    result = aug.augment(img.copy())
    assert isinstance(result, np.ndarray)
    assert result.shape == img.shape
