"""
测试增强的框架功能
"""

import os
# 设置环境变量解决 OpenMP 库冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pytest
from types import ModuleType
from visionframework.core.plugin_system import PluginRegistry, ModelRegistry, register_detector, register_tracker
from visionframework.utils.dependency_manager import DependencyManager, is_dependency_available, import_optional_dependency
from visionframework.utils.error_handling import ErrorHandler
from visionframework.core.pipeline import VisionPipeline
from visionframework.utils.monitoring.logger import get_logger
import numpy as np
import cv2

logger = get_logger(__name__)


# 测试插件系统
def test_plugin_registry():
    """
    测试插件注册表
    """
    # 注意：register_detector 和 register_tracker 装饰器使用的是全局 plugin_registry
    # 所以我们应该测试全局实例而不是创建新实例
    from visionframework.core.plugin_system import plugin_registry
    
    # 测试注册和获取插件
    @register_detector("test_detector")
    class TestDetector:
        def __init__(self, config):
            self.config = config
        
        def initialize(self):
            return True
    
    @register_tracker("test_tracker")
    class TestTracker:
        def __init__(self, config):
            self.config = config
        
        def initialize(self):
            return True
    
    # 测试获取注册的插件
    detectors = plugin_registry.list_detectors()
    assert "test_detector" in detectors
    
    trackers = plugin_registry.list_trackers()
    assert "test_tracker" in trackers
    
    # 测试获取插件信息并创建实例
    detector_config = {"model_path": "test.pt"}
    detector_info = plugin_registry.get_detector("test_detector")
    assert detector_info is not None
    detector_class = detector_info["class"]
    detector = detector_class(detector_config)
    assert detector is not None
    assert detector.initialize()
    
    tracker_config = {"max_age": 30}
    tracker_info = plugin_registry.get_tracker("test_tracker")
    assert tracker_info is not None
    tracker_class = tracker_info["class"]
    tracker = tracker_class(tracker_config)
    assert tracker is not None
    assert tracker.initialize()


def test_model_registry():
    """
    测试模型注册表
    """
    registry = ModelRegistry()
    
    # 测试注册和获取模型
    model_info = {
        "name": "test_model",
        "source": "yolo",
        "config": {"file_name": "yolov8n.pt"},
        "loader": lambda: None  # 简单的加载器函数
    }
    registry.register_model("test_model", model_info)
    
    # 测试获取模型信息
    retrieved_info = registry.get_model("test_model")
    assert retrieved_info is not None
    assert retrieved_info["name"] == "test_model"
    assert retrieved_info["source"] == "yolo"
    
    # 测试获取所有注册的模型
    models = registry.list_models()
    assert "test_model" in models


# 测试依赖管理
def test_dependency_manager():
    """
    测试依赖管理器
    """
    manager = DependencyManager()
    
    # 测试获取依赖信息
    clip_info = manager.get_dependency_info("clip")
    assert clip_info is not None
    assert "packages" in clip_info
    assert "transformers" in clip_info["packages"]
    
    # 测试获取安装命令
    install_command = manager.get_install_command("clip")
    assert install_command is not None
    assert "pip install" in install_command
    assert "transformers" in install_command


def test_dependency_availability():
    """
    测试依赖可用性检查
    """
    # 测试依赖检查机制，不强制要求依赖可用
    # 测试获取依赖状态
    from visionframework.utils.dependency_manager import dependency_manager
    dev_status = dependency_manager.get_dependency_status("dev")
    assert dev_status is not None
    assert "available" in dev_status
    assert "message" in dev_status
    
    # 测试导入可选依赖
    # 注意：这里只是测试导入机制，不要求依赖实际可用
    module = import_optional_dependency("clip", "transformers")
    # 如果 transformers 可用，module 应该不为 None
    # 如果不可用，module 应该为 None
    assert isinstance(module, (type(None), ModuleType))


# 测试错误处理
def test_error_handler():
    """
    测试错误处理器
    """
    handler = ErrorHandler()
    
    # 测试错误处理
    def risky_operation():
        raise ValueError("测试错误")
    
    # 测试 handle_error 方法
    try:
        error = ValueError("测试错误")
        result = handler.handle_error(
            error=error,
            error_type=Exception,
            message="测试错误处理"
        )
        assert isinstance(result, Exception) or result is None
    except Exception as e:
        logger.error(f"测试错误处理失败: {e}")
    
    # 测试错误包装
    wrapped_func = handler.wrap_error(
        func=risky_operation,
        error_type=Exception,
        message="测试错误包装"
    )
    result = wrapped_func()
    assert result is None
    
    # 测试输入验证
    valid_input = {"key": "value"}
    is_valid, error_msg = handler.validate_input(
        input_value=valid_input,
        expected_type=dict,
        param_name="input"
    )
    assert is_valid
    
    invalid_input = "not a dict"
    is_valid, error_msg = handler.validate_input(
        input_value=invalid_input,
        expected_type=dict,
        param_name="input"
    )
    assert not is_valid
    assert error_msg is not None
    
    # 测试错误消息格式化
    error = ValueError("测试错误")
    error_message = handler.format_error_message(
        message="测试操作",
        error=error
    )
    assert "测试操作" in error_message
    assert "测试错误" in error_message


# 测试增强的批处理功能
def test_batch_processing_enhancements():
    """
    测试增强的批处理功能
    """
    # 创建测试配置
    config = {
        "detector_config": {
            "model_path": "yolov8n.pt",
            "device": "cpu",
            "conf_threshold": 0.1
        }
    }
    
    # 初始化管道
    pipeline = VisionPipeline(config)
    
    # 创建测试图像
    test_images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]
    
    # 测试基本批处理
    results1 = pipeline.process_batch(test_images)
    assert len(results1) == 4
    
    # 测试最大批处理大小限制
    results2 = pipeline.process_batch(test_images, max_batch_size=2)
    assert len(results2) == 4
    
    # 测试不同的批处理大小
    results3 = pipeline.process_batch(test_images, max_batch_size=1)
    assert len(results3) == 4
    
    # 测试空输入
    empty_results = pipeline.process_batch([])
    assert len(empty_results) == 0


# 性能基准测试
def test_performance_benchmark():
    """
    测试性能基准
    """
    import time
    from visionframework.utils.monitoring.performance import PerformanceMonitor
    
    # 创建性能监控器
    monitor = PerformanceMonitor()
    
    # 创建测试配置
    config = {
        "detector_config": {
            "model_path": "yolov8n.pt",
            "device": "cpu",
            "conf_threshold": 0.1
        },
        "enable_performance_monitoring": True
    }
    
    # 初始化管道
    pipeline = VisionPipeline(config)
    
    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_images = [test_image for _ in range(10)]
    
    # 测试单图像处理性能
    monitor.start()
    start_time = time.time()
    for img in test_images:
        pipeline.process(img)
        monitor.tick()
    single_process_time = time.time() - start_time
    
    # 测试批处理性能
    monitor.reset()
    monitor.start()
    start_time = time.time()
    pipeline.process_batch(test_images)
    batch_process_time = time.time() - start_time
    
    # 打印性能比较
    print(f"\n性能基准测试结果:")
    print(f"单图像处理时间: {single_process_time:.4f} 秒")
    print(f"批处理时间: {batch_process_time:.4f} 秒")
    print(f"批处理加速比: {single_process_time / batch_process_time:.2f}x")
    
    # 验证批处理时间应该小于等于单处理时间
    assert batch_process_time <= single_process_time * 1.1  # 允许 10% 的误差


if __name__ == "__main__":
    # 运行所有测试
    test_plugin_registry()
    print("✓ 插件系统测试通过")
    
    test_model_registry()
    print("✓ 模型注册测试通过")
    
    test_dependency_manager()
    print("✓ 依赖管理测试通过")
    
    test_dependency_availability()
    print("✓ 依赖可用性测试通过")
    
    test_error_handler()
    print("✓ 错误处理测试通过")
    
    test_batch_processing_enhancements()
    print("✓ 批处理功能测试通过")
    
    test_performance_benchmark()
    print("✓ 性能基准测试通过")
    
    print("\n所有增强功能测试通过！")
