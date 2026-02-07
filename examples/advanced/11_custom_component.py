"""
11 - 自定义组件
===============
演示如何创建自定义检测器和处理器，并通过插件系统注册使用。
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np

from visionframework import (
    BaseDetector, BaseProcessor, VisionPipeline,
    Detection, Visualizer,
    register_detector, register_processor,
)


# ── 自定义检测器：颜色阈值检测 ──

class CustomDetector(BaseDetector):
    def __init__(self, config: dict):
        super().__init__(config)
        self.lower_color = config.get("lower_color", [0, 0, 100])
        self.upper_color = config.get("upper_color", [100, 100, 255])
        self.min_area = config.get("min_area", 100)

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def detect(self, image: np.ndarray) -> list:
        if not self._initialized:
            return []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array(self.lower_color, dtype=np.uint8)
        upper = np.array(self.upper_color, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            if cv2.contourArea(c) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append(Detection(
                bbox=(x, y, x + w, y + h),
                confidence=0.8,
                class_id=0,
                class_name="custom_object",
            ))
        return detections


# ── 自定义处理器：目标大小过滤 ──

class CustomProcessor(BaseProcessor):
    def __init__(self, config: dict):
        super().__init__(config)
        self.min_width = config.get("min_width", 50)
        self.min_height = config.get("min_height", 50)

    def initialize(self) -> bool:
        self._initialized = True
        return True

    def process(self, image: np.ndarray, detections: list = None) -> list:
        if not self._initialized or detections is None:
            return []
        return [
            d for d in detections
            if (d.bbox[2] - d.bbox[0]) >= self.min_width
            and (d.bbox[3] - d.bbox[1]) >= self.min_height
        ]


# ── 注册到插件系统 ──

@register_detector("custom")
def _factory_custom_detector(config: dict) -> BaseDetector:
    return CustomDetector(config)

@register_processor("custom")
def _factory_custom_processor(config: dict) -> BaseProcessor:
    return CustomProcessor(config)


# ── 测试 ──

def main():
    # 1. 直接使用自定义检测器
    detector = CustomDetector({
        "model_path": "dummy", "device": "cpu",
        "lower_color": [0, 0, 100], "upper_color": [100, 100, 255],
        "min_area": 100,
    })
    detector.initialize()

    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 0, 255), -1)
    cv2.rectangle(test_image, (300, 150), (400, 250), (0, 0, 255), -1)
    cv2.rectangle(test_image, (50, 300), (80, 330), (0, 0, 255), -1)  # 小方块

    detections = detector.detect(test_image)
    print(f"自定义检测器: {len(detections)} 个目标")

    # 2. 在 Pipeline 中使用
    pipeline = VisionPipeline({
        "detector_config": {
            "model_path": "dummy", "model_type": "custom", "device": "cpu",
            "lower_color": [0, 0, 100], "upper_color": [100, 100, 255],
            "min_area": 100,
        },
        "enable_tracking": True,
        "tracker_config": {"tracker_type": "bytetrack"},
    })
    pipeline.initialize()

    result = pipeline.process(test_image)
    print(f"管道: {len(result.get('detections', []))} 检测, "
          f"{len(result.get('tracks', []))} 轨迹")

    # 可视化
    vis = Visualizer()
    vis_image = vis.draw_results(test_image.copy(),
                                  detections=result.get("detections", []),
                                  tracks=result.get("tracks", []))
    cv2.imshow("Custom Components", vis_image)
    print("按任意键结束...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
