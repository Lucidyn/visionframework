"""
11_custom_component.py

自定义组件示例：
- 创建自定义检测器
- 创建自定义处理器
- 注册自定义组件
- 在管道中使用自定义组件

注意：
- 此示例展示了如何扩展 VisionFramework 的功能
"""

import os

# 解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np

from visionframework import (
    BaseDetector, BaseProcessor, VisionPipeline,
    Detection, register_detector, register_processor,
    create_detector, create_pipeline, Visualizer
)


class CustomDetector(BaseDetector):
    """
    自定义检测器示例。
    这个检测器实现了一个简单的颜色阈值检测。
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.lower_color = config.get("lower_color", [0, 0, 100])  # 默认检测红色
        self.upper_color = config.get("upper_color", [100, 100, 255])
        self.min_area = config.get("min_area", 100)
    
    def initialize(self) -> bool:
        """
        初始化检测器。
        对于这个简单的检测器，初始化总是成功的。
        """
        print("CustomDetector 初始化成功!")
        self._initialized = True
        return True
    
    def detect(self, image: np.ndarray) -> list:
        """
        执行检测。
        使用颜色阈值和轮廓检测来检测目标。
        """
        if not self._initialized:
            return []
        
        # 转换为 HSV 颜色空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 创建颜色掩码
        lower = np.array(self.lower_color, dtype=np.uint8)
        upper = np.array(self.upper_color, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        
        # 形态学操作
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤小面积轮廓
            if area < self.min_area:
                continue
            
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, x + w, y + h)
            
            # 创建 Detection 对象
            detection = Detection(
                bbox=bbox,
                confidence=0.8,  # 固定置信度
                class_id=0,  # 固定类别 ID
                class_name="custom_object"  # 固定类别名称
            )
            
            detections.append(detection)
        
        return detections


class CustomProcessor(BaseProcessor):
    """
    自定义处理器示例。
    这个处理器实现了一个简单的目标大小过滤器。
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.min_width = config.get("min_width", 50)
        self.min_height = config.get("min_height", 50)
    
    def initialize(self) -> bool:
        """
        初始化处理器。
        对于这个简单的处理器，初始化总是成功的。
        """
        print("CustomProcessor 初始化成功!")
        self._initialized = True
        return True
    
    def process(self, image: np.ndarray, detections: list = None) -> list:
        """
        执行处理。
        过滤掉太小的目标。
        """
        if not self._initialized or detections is None:
            return []
        
        filtered_detections = []
        
        for detection in detections:
            # 获取边界框
            x1, y1, x2, y2 = detection.bbox
            width = x2 - x1
            height = y2 - y1
            
            # 过滤小目标
            if width >= self.min_width and height >= self.min_height:
                filtered_detections.append(detection)
        
        return filtered_detections


# 注册自定义组件
@register_detector("custom")
def create_custom_detector(config: dict) -> BaseDetector:
    """
    创建自定义检测器的工厂函数。
    这个函数会被 VisionFramework 用于创建自定义检测器实例。
    """
    return CustomDetector(config)


@register_processor("custom")
def create_custom_processor(config: dict) -> BaseProcessor:
    """
    创建自定义处理器的工厂函数。
    这个函数会被 VisionFramework 用于创建自定义处理器实例。
    """
    return CustomProcessor(config)


def test_custom_detector():
    """
    测试自定义检测器。
    """
    print("测试自定义检测器...")
    
    # 创建自定义检测器配置
    config = {
        "model_path": "dummy_path",  # 自定义检测器不使用模型路径
        "device": "cpu",
        "lower_color": [0, 0, 100],  # 检测红色
        "upper_color": [100, 100, 255],
        "min_area": 100,
    }
    
    # 使用 create_detector 创建自定义检测器
    detector = create_detector(
        model_path="dummy_path",
        model_type="custom",  # 指定使用自定义检测器
        device="cpu",
        **config
    )
    
    # 初始化检测器
    if not detector.initialize():
        print("自定义检测器初始化失败!")
        return
    
    # 创建测试图像（包含红色方块）
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # 绘制红色方块
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 0, 255), -1)
    cv2.rectangle(test_image, (300, 150), (400, 250), (0, 0, 255), -1)
    cv2.rectangle(test_image, (50, 300), (80, 330), (0, 0, 255), -1)  # 小方块，应该被过滤掉
    
    # 执行检测
    detections = detector.detect(test_image)
    print(f"自定义检测器检测到 {len(detections)} 个目标")
    
    # 可视化结果
    visualizer = Visualizer()
    vis_image = visualizer.draw_detections(test_image.copy(), detections)
    
    cv2.imshow("Custom Detector Results", vis_image)
    print("按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_custom_processor():
    """
    测试自定义处理器。
    """
    print("测试自定义处理器...")
    
    # 创建自定义处理器配置
    config = {
        "min_width": 50,
        "min_height": 50,
    }
    
    # 创建自定义处理器
    processor = CustomProcessor(config)
    
    # 初始化处理器
    if not processor.initialize():
        print("自定义处理器初始化失败!")
        return
    
    # 创建测试检测结果
    test_detections = [
        Detection(
            bbox=(100, 100, 200, 200),  # 100x100, 应该通过
            confidence=0.8,
            class_id=0,
            class_name="object"
        ),
        Detection(
            bbox=(300, 150, 320, 170),  # 20x20, 应该被过滤
            confidence=0.8,
            class_id=0,
            class_name="object"
        ),
    ]
    
    # 执行处理
    filtered_detections = processor.process(np.zeros((480, 640, 3), dtype=np.uint8), test_detections)
    print(f"自定义处理器过滤后剩余 {len(filtered_detections)} 个目标")


def test_custom_components_in_pipeline():
    """
    测试在管道中使用自定义组件。
    """
    print("测试在管道中使用自定义组件...")
    
    # 创建管道配置
    pipeline_config = {
        "detector_config": {
            "model_path": "dummy_path",
            "model_type": "custom",  # 使用自定义检测器
            "device": "cpu",
            "lower_color": [0, 0, 100],  # 检测红色
            "upper_color": [100, 100, 255],
            "min_area": 100,
        },
        "enable_tracking": True,
        "tracker_config": {
            "tracker_type": "bytetrack",
        },
        "custom_processors": [
            {
                "type": "custom",  # 使用自定义处理器
                "min_width": 50,
                "min_height": 50,
            }
        ],
    }
    
    # 创建管道
    pipeline = create_pipeline(
        detector_config=pipeline_config["detector_config"],
        enable_tracking=pipeline_config["enable_tracking"],
        tracker_config=pipeline_config["tracker_config"]
    )
    
    # 初始化管道
    if not pipeline.initialize():
        print("管道初始化失败!")
        return
    
    # 创建测试图像（包含红色方块）
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    # 绘制红色方块
    cv2.rectangle(test_image, (100, 100), (200, 200), (0, 0, 255), -1)
    cv2.rectangle(test_image, (300, 150), (400, 250), (0, 0, 255), -1)
    cv2.rectangle(test_image, (50, 300), (80, 330), (0, 0, 255), -1)  # 小方块，应该被过滤掉
    
    # 执行处理
    result = pipeline.process(test_image)
    detections = result.get("detections", [])
    tracks = result.get("tracks", [])
    
    print(f"管道处理后检测到 {len(detections)} 个目标")
    print(f"管道处理后跟踪到 {len(tracks)} 条轨迹")
    
    # 可视化结果
    visualizer = Visualizer()
    vis_image = visualizer.draw_results(
        test_image.copy(),
        detections=detections,
        tracks=tracks
    )
    
    cv2.imshow("Pipeline with Custom Components Results", vis_image)
    print("按任意键结束...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """
    主函数。
    """
    # 测试自定义检测器
    test_custom_detector()
    
    # 测试自定义处理器
    test_custom_processor()
    
    # 测试在管道中使用自定义组件
    test_custom_components_in_pipeline()


if __name__ == "__main__":
    main()
