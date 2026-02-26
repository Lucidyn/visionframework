"""
检测器与跟踪器输入验证测试。

验证新添加的输入验证功能是否正常工作。
"""

import numpy as np
import pytest

from visionframework import BaseDetector, BaseTracker, Detection


class DummyDetector(BaseDetector):
    """用于测试的虚拟检测器。"""

    def initialize(self) -> bool:
        self.is_initialized = True
        return True

    def detect(self, image, categories=None):
        self._validate_image(image)
        return []


class DummyTracker(BaseTracker):
    """用于测试的虚拟跟踪器。"""

    def update(self, detections, frame=None):
        detections = self._validate_detections(detections)
        return []


# ===== 检测器图像验证 =====

def test_detector_validate_image_valid_3d():
    """有效的 3D 图像应通过验证。"""
    detector = DummyDetector({})
    detector.initialize()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    result = detector.detect(image)
    assert isinstance(result, list)


def test_detector_validate_image_valid_2d():
    """有效的 2D 灰度图像应通过验证。"""
    detector = DummyDetector({})
    detector.initialize()
    image = np.zeros((480, 640), dtype=np.uint8)
    result = detector.detect(image)
    assert isinstance(result, list)


def test_detector_validate_image_not_ndarray():
    """非 ndarray 应抛出 TypeError。"""
    detector = DummyDetector({})
    detector.initialize()
    with pytest.raises(TypeError, match="Expected np.ndarray"):
        detector.detect([1, 2, 3])


def test_detector_validate_image_wrong_ndim():
    """错误的维度应抛出 ValueError。"""
    detector = DummyDetector({})
    detector.initialize()
    with pytest.raises(ValueError, match="must be 2D.*or 3D"):
        detector.detect(np.zeros(100))
    with pytest.raises(ValueError, match="must be 2D.*or 3D"):
        detector.detect(np.zeros((10, 480, 640, 3)))


def test_detector_validate_image_empty():
    """空图像应抛出 ValueError。"""
    detector = DummyDetector({})
    detector.initialize()
    with pytest.raises(ValueError, match="empty.*zero pixels"):
        detector.detect(np.zeros((0, 0, 3)))


# ===== 跟踪器检测列表验证 =====

def test_tracker_validate_detections_valid_list():
    """有效的检测列表应通过验证。"""
    tracker = DummyTracker({})
    tracker.initialize()
    detections = [
        Detection(bbox=(10, 20, 30, 40), confidence=0.9, class_id=0, class_name="person"),
        Detection(bbox=(50, 60, 70, 80), confidence=0.8, class_id=1, class_name="car"),
    ]
    result = tracker.update(detections)
    assert isinstance(result, list)


def test_tracker_validate_detections_empty_list():
    """空列表应通过验证。"""
    tracker = DummyTracker({})
    tracker.initialize()
    result = tracker.update([])
    assert isinstance(result, list)


def test_tracker_validate_detections_none():
    """None 应被转换为空列表。"""
    tracker = DummyTracker({})
    tracker.initialize()
    result = tracker.update(None)
    assert isinstance(result, list)


def test_tracker_validate_detections_tuple():
    """元组也应被接受。"""
    tracker = DummyTracker({})
    tracker.initialize()
    detections = (
        Detection(bbox=(10, 20, 30, 40), confidence=0.9, class_id=0, class_name="person"),
    )
    result = tracker.update(detections)
    assert isinstance(result, list)


def test_tracker_validate_detections_not_iterable():
    """非可迭代对象应抛出 TypeError。"""
    tracker = DummyTracker({})
    tracker.initialize()
    with pytest.raises(TypeError, match="must be a list/tuple"):
        tracker.update(123)


def test_tracker_validate_detections_generator():
    """生成器应被转换为列表。"""
    tracker = DummyTracker({})
    tracker.initialize()

    def gen():
        yield Detection(bbox=(10, 20, 30, 40), confidence=0.9, class_id=0, class_name="person")

    result = tracker.update(gen())
    assert isinstance(result, list)
