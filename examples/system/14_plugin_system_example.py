#!/usr/bin/env python3
"""
插件系统使用示例

本示例演示如何使用 Vision Framework 的插件系统注册和使用自定义组件。
包括：
1. 注册自定义检测器
2. 注册自定义跟踪器
3. 注册自定义模型
4. 使用自定义组件创建管道
5. 测试自定义组件的功能
"""

import os
import sys
import cv2
import numpy as np

# 修复OpenMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionframework import (
    register_detector, register_tracker, register_model,
    plugin_registry, model_registry,
    VisionPipeline, Detection
)


# 1. 注册自定义检测器
@register_detector("simple_detector", author="Vision Framework", version="1.0")
class SimpleDetector:
    """简单的自定义检测器示例"""
    
    def __init__(self, config):
        """
        初始化检测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.conf_threshold = config.get("conf_threshold", 0.5)
        self.class_names = config.get("class_names", ["object", "person", "car"])
        print(f"初始化简单检测器，置信度阈值: {self.conf_threshold}")
    
    def initialize(self):
        """
        初始化检测器
        
        Returns:
            bool: 初始化是否成功
        """
        print("简单检测器初始化成功")
        return True
    
    def detect(self, image):
        """
        检测图像中的目标
        
        Args:
            image: 输入图像
            categories: 要检测的类别列表（可选）
            **kwargs: 额外参数
        
        Returns:
            List[Detection]: 检测结果列表
        """
        detections = []
        
        # 模拟检测结果
        height, width = image.shape[:2]
        
        # 检测中心的一个目标
        center_x, center_y = width // 2, height // 2
        bbox = (center_x - 50, center_y - 50, center_x + 50, center_y + 50)
        confidence = 0.9
        class_id = 0
        class_name = self.class_names[class_id]
        
        detection = Detection(
            bbox=bbox,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name
        )
        detections.append(detection)
        
        # 检测左上角的一个目标
        bbox2 = (50, 50, 150, 150)
        confidence2 = 0.8
        class_id2 = 1
        class_name2 = self.class_names[class_id2]
        
        detection2 = Detection(
            bbox=bbox2,
            confidence=confidence2,
            class_id=class_id2,
            class_name=class_name2
        )
        detections.append(detection2)
        
        print(f"简单检测器检测到 {len(detections)} 个目标")
        return detections
    
    def detect_batch(self, images):
        """
        批量检测多张图像
        
        Args:
            images: 图像列表
            **kwargs: 额外参数
        
        Returns:
            List[List[Detection]]: 检测结果列表
        """
        results = []
        for image in images:
            results.append(self.detect(image))
        return results
    
    def cleanup(self):
        """
        清理资源
        """
        print("简单检测器清理资源")


# 2. 注册自定义跟踪器
@register_tracker("simple_tracker", author="Vision Framework", version="1.0")
class SimpleTracker:
    """简单的自定义跟踪器示例"""
    
    def __init__(self, config):
        """
        初始化跟踪器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.max_age = config.get("max_age", 30)
        self.iou_threshold = config.get("iou_threshold", 0.3)
        self.next_track_id = 1
        self.tracks = {}
        print(f"初始化简单跟踪器，最大年龄: {self.max_age}, IoU阈值: {self.iou_threshold}")
    
    def initialize(self):
        """
        初始化跟踪器
        
        Returns:
            bool: 初始化是否成功
        """
        print("简单跟踪器初始化成功")
        return True
    
    def update(self, detections, frame=None):
        """
        更新跟踪
        
        Args:
            detections: 检测结果列表
            frame: 当前帧（可选）
        
        Returns:
            List[Detection]: 更新后的检测结果列表
        """
        # 为每个检测结果分配跟踪ID
        for detection in detections:
            if not hasattr(detection, 'track_id') or detection.track_id is None:
                detection.track_id = self.next_track_id
                self.next_track_id += 1
        
        print(f"简单跟踪器更新了 {len(detections)} 个目标")
        return detections
    
    def reset(self):
        """
        重置跟踪器
        """
        self.next_track_id = 1
        self.tracks = {}
        print("简单跟踪器重置")


# 3. 注册自定义模型
@register_model("simple_model", author="Vision Framework", version="1.0")
def simple_model_loader():
    """
    简单的模型加载函数
    
    Returns:
        Any: 加载的模型
    """
    print("加载简单模型")
    # 模拟模型加载
    class SimpleModel:
        def predict(self, x):
            return np.random.rand(len(x), 1)
    
    return SimpleModel()


def main():
    """主函数"""
    print("=== 插件系统示例 ===")
    
    # 4. 列出所有注册的组件
    print("\n1. 列出所有注册的组件:")
    print("注册的检测器:", plugin_registry.list_detectors())
    print("注册的跟踪器:", plugin_registry.list_trackers())
    
    # 5. 使用自定义组件创建管道
    print("\n2. 使用自定义组件创建管道:")
    config = {
        "detector_config": {
            "model_path": "simple_detector",  # 使用自定义检测器
            "conf_threshold": 0.3
        },
        "tracker_config": {
            "tracker_type": "simple_tracker",  # 使用自定义跟踪器
            "max_age": 20,
            "iou_threshold": 0.4
        },
        "enable_tracking": True
    }
    
    pipeline = VisionPipeline(config)
    print("管道创建成功")
    
    # 6. 初始化管道
    print("\n3. 初始化管道:")
    pipeline.initialize()
    
    # 7. 测试自定义组件
    print("\n4. 测试自定义组件:")
    
    # 创建测试图像
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (300, 300), (255, 255, 255), -1)
    cv2.rectangle(image, (400, 200), (500, 300), (255, 0, 0), -1)
    
    # 处理图像
    results = pipeline.process(image)
    print(f"检测结果数量: {len(results['detections'])}")
    
    # 打印检测结果
    for i, detection in enumerate(results['detections']):
        print(f"  检测目标 {i+1}:")
        print(f"    类别: {detection.class_name}")
        print(f"    置信度: {detection.confidence:.2f}")
        print(f"    边界框: {detection.bbox}")
        print(f"    跟踪ID: {detection.track_id}")
    
    # 8. 测试模型注册表
    print("\n5. 测试模型注册表:")
    
    # 注册模型
    model_info = {
        "name": "test_model",
        "source": "custom",
        "config": {"file_path": "test_model.pt"},
        "loader": simple_model_loader
    }
    model_registry.register_model("test_model", model_info)
    
    # 列出所有注册的模型
    print("注册的模型:", model_registry.list_models())
    
    # 加载模型
    model = model_registry.load_model("test_model")
    print(f"模型加载成功: {model is not None}")
    
    # 测试模型
    if model:
        test_input = np.array([[1.0], [2.0], [3.0]])
        prediction = model.predict(test_input)
        print(f"模型预测结果: {prediction}")
    
    # 9. 测试插件发现机制
    print("\n6. 测试插件发现机制:")
    print("添加插件路径:")
    # 这里可以添加自定义插件路径
    # plugin_registry.add_plugin_path("path/to/plugins")
    
    print("加载插件:")
    # plugin_registry.load_all_plugins()
    
    print("\n=== 插件系统示例完成 ===")


if __name__ == "__main__":
    main()
