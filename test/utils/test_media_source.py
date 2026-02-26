"""
iter_frames 工具测试。

覆盖：numpy 数组、图像文件、路径列表、文件夹。
"""

import os
import tempfile
import cv2
import numpy as np
import pytest

from visionframework import iter_frames


def _make_image(h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# numpy 数组输入
# ---------------------------------------------------------------------------

def test_iter_frames_numpy_array():
    img = _make_image()
    frames = list(iter_frames(img))
    assert len(frames) == 1
    frame, meta = frames[0]
    assert isinstance(frame, np.ndarray)
    assert frame.shape == img.shape
    assert isinstance(meta, dict)


def test_iter_frames_numpy_array_meta_keys():
    img = _make_image()
    frames = list(iter_frames(img))
    _, meta = frames[0]
    assert isinstance(meta, dict)
    assert len(meta) > 0


# ---------------------------------------------------------------------------
# 图像文件输入
# ---------------------------------------------------------------------------

def test_iter_frames_image_file():
    img = _make_image()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        path = f.name

    try:
        cv2.imwrite(path, img)
        frames = list(iter_frames(path))
        assert len(frames) == 1
        frame, meta = frames[0]
        assert isinstance(frame, np.ndarray)
        assert frame.shape[2] == 3
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# 路径列表输入
# ---------------------------------------------------------------------------

def test_iter_frames_list_of_paths():
    images = [_make_image() for _ in range(3)]
    paths = []
    try:
        for i, img in enumerate(images):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                path = f.name
            cv2.imwrite(path, img)
            paths.append(path)

        frames = list(iter_frames(paths))
        assert len(frames) == 3
        for frame, meta in frames:
            assert isinstance(frame, np.ndarray)
    finally:
        for p in paths:
            if os.path.exists(p):
                os.unlink(p)


# ---------------------------------------------------------------------------
# 文件夹输入
# ---------------------------------------------------------------------------

def test_iter_frames_folder():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            img = _make_image()
            cv2.imwrite(os.path.join(tmpdir, f"img_{i:03d}.jpg"), img)

        frames = list(iter_frames(tmpdir))
        assert len(frames) == 3
        for frame, meta in frames:
            assert isinstance(frame, np.ndarray)


# ---------------------------------------------------------------------------
# 边界情况
# ---------------------------------------------------------------------------

def test_iter_frames_empty_list():
    frames = list(iter_frames([]))
    assert frames == []


def test_iter_frames_list_of_arrays():
    images = [_make_image() for _ in range(2)]
    frames = list(iter_frames(images))
    assert len(frames) == 2
    for frame, meta in frames:
        assert isinstance(frame, np.ndarray)
