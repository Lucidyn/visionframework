"""
Tests for processors (CLIPExtractor, ReIDExtractor).

CLIP 依赖 transformers，属于可选依赖。
测试策略：
- 构造实例不需要依赖 -> 总是测试
- 需要模型的方法 -> 尝试 initialize，若依赖缺失则 skip
"""

import pytest
import numpy as np

from visionframework import CLIPExtractor, ReIDExtractor


def _make_dummy_image(h: int = 224, w: int = 224) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ================================================================
# CLIPExtractor
# ================================================================

def test_clip_creation() -> None:
    """CLIPExtractor 构造不需要依赖，总是成功。"""
    clip = CLIPExtractor(model_name="openai/clip-vit-base-patch32", device="cpu")
    assert isinstance(clip, CLIPExtractor)
    assert clip.is_initialized is False


def test_clip_default_params() -> None:
    """默认参数检查。"""
    clip = CLIPExtractor()
    assert clip.model_name == "openai/clip-vit-base-patch32"
    assert clip.device == "cpu"
    assert clip.use_fp16 is False


def _try_init_clip():
    """尝试初始化 CLIP，如果依赖缺失则 skip。"""
    clip = CLIPExtractor(model_name="openai/clip-vit-base-patch32", device="cpu")
    try:
        clip.initialize()
    except (ImportError, RuntimeError, ValueError) as e:
        pytest.skip(f"CLIP init unavailable: {e}")
    return clip


def test_clip_encode_image() -> None:
    """encode_image 应返回归一化的 numpy 向量。"""
    clip = _try_init_clip()
    img = _make_dummy_image()
    emb = clip.encode_image(img)
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 2  # (1, D)
    assert emb.shape[0] == 1
    # 检查归一化
    norm = np.linalg.norm(emb[0])
    assert abs(norm - 1.0) < 0.01, f"Embedding not normalized, norm={norm}"


def test_clip_encode_text() -> None:
    """encode_text 应返回归一化的 numpy 向量。"""
    clip = _try_init_clip()
    texts = ["a photo of a cat", "a photo of a dog"]
    emb = clip.encode_text(texts)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] == 2  # 两个文本
    # 检查归一化
    for i in range(emb.shape[0]):
        norm = np.linalg.norm(emb[i])
        assert abs(norm - 1.0) < 0.01


def test_clip_image_text_similarity() -> None:
    """image_text_similarity 应返回 (num_images, num_texts) 矩阵。"""
    clip = _try_init_clip()
    img = _make_dummy_image()
    texts = ["cat", "dog", "car"]
    sim = clip.image_text_similarity(img, texts)
    assert isinstance(sim, np.ndarray)
    assert sim.shape == (1, 3)


def test_clip_zero_shot_classify() -> None:
    """zero_shot_classify 应返回与标签数量一致的分数列表。"""
    clip = _try_init_clip()
    img = _make_dummy_image()
    labels = ["person", "car", "tree"]
    scores = clip.zero_shot_classify(img, labels)
    assert isinstance(scores, list)
    assert len(scores) == len(labels)
    for s in scores:
        assert isinstance(s, float)


def test_clip_extract_image() -> None:
    """extract(image) 应等同于 encode_image。"""
    clip = _try_init_clip()
    img = _make_dummy_image()
    result = clip.extract(img)
    assert isinstance(result, np.ndarray)


def test_clip_extract_text() -> None:
    """extract(text_list) 应等同于 encode_text。"""
    clip = _try_init_clip()
    result = clip.extract(["hello world"])
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 1


def test_clip_cleanup() -> None:
    """cleanup 不应抛异常。"""
    clip = CLIPExtractor(model_name="openai/clip-vit-base-patch32", device="cpu")
    clip.cleanup()  # 未初始化时也不应报错
    assert clip.is_initialized is False


def test_clip_batch_encode_image() -> None:
    """encode_image 支持 batch。"""
    clip = _try_init_clip()
    imgs = [_make_dummy_image() for _ in range(3)]
    emb = clip.encode_image(imgs)
    assert isinstance(emb, np.ndarray)
    assert emb.shape[0] == 3


# ================================================================
# ReIDExtractor
# ================================================================

def test_reid_creation() -> None:
    """ReIDExtractor 构造不需要依赖，总是成功。"""
    reid = ReIDExtractor(model_name="resnet50", device="cpu", model_path="nonexistent.pt")
    assert isinstance(reid, ReIDExtractor)


def test_reid_default_not_initialized() -> None:
    """构造后未初始化。"""
    reid = ReIDExtractor(model_name="resnet50", device="cpu", model_path="nonexistent.pt")
    assert reid.is_initialized is False
