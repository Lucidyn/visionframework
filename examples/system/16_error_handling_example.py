#!/usr/bin/env python3
"""
错误处理使用示例

本示例演示如何使用 Vision Framework 的统一错误处理功能。
包括：
1. 使用 ErrorHandler 处理错误
2. 包装函数以捕获和处理错误
3. 输入验证
4. 错误消息格式化
5. 使用自定义异常类
6. 在管道中使用错误处理
7. 错误处理最佳实践
"""

import os
import sys
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionframework.utils.error_handling import ErrorHandler
from visionframework.core.pipeline import VisionPipeline
from visionframework.exceptions import (
    VisionFrameworkError,
    ModelError,
    ConfigurationError,
    DeviceError,
    VideoError,
    ProcessingError,
    DataFormatError
)

def test_basic_error_handling():
    """
    测试基本的错误处理功能
    """
    print("=== 测试基本错误处理功能 ===")
    
    # 1. 创建错误处理器
    print("\n1. 创建错误处理器:")
    handler = ErrorHandler()
    print("错误处理器创建成功")
    
    # 2. 处理错误
    print("\n2. 处理错误:")
    try:
        # 模拟错误
        error = ValueError("测试错误")
        result = handler.handle_error(
            error=error,
            error_type=Exception,
            message="测试错误处理"
        )
        print(f"错误处理结果: {result}")
    except Exception as e:
        print(f"错误处理失败: {e}")
    
    # 3. 测试错误抛出
    print("\n3. 测试错误抛出:")
    try:
        error = ValueError("测试错误")
        result = handler.handle_error(
            error=error,
            error_type=RuntimeError,
            message="测试错误抛出",
            raise_error=True
        )
        print(f"错误处理结果: {result}")  # 这里不会执行
    except RuntimeError as e:
        print(f"错误抛出成功: {e}")
    
    # 4. 测试错误消息格式化
    print("\n4. 测试错误消息格式化:")
    error = ValueError("测试错误")
    error_message = handler.format_error_message(
        message="测试操作",
        error=error,
        context={"param": "value", "stage": "processing"}
    )
    print(f"格式化错误消息: {error_message}")
    
    return handler

def test_error_wrapping():
    """
    测试错误包装功能
    """
    print("\n=== 测试错误包装功能 ===")
    
    handler = ErrorHandler()
    
    # 1. 创建可能出错的函数
    print("\n1. 创建可能出错的函数:")
    def risky_operation():
        raise ValueError("操作失败")
    
    def safe_operation():
        return "操作成功"
    
    # 2. 包装错误函数
    print("\n2. 包装错误函数:")
    wrapped_risky = handler.wrap_error(
        func=risky_operation,
        error_type=RuntimeError,
        message="风险操作错误",
        default_return="默认值"
    )
    
    wrapped_safe = handler.wrap_error(
        func=safe_operation,
        error_type=RuntimeError,
        message="安全操作错误"
    )
    
    # 3. 测试包装函数
    print("\n3. 测试包装函数:")
    result = wrapped_risky()
    print(f"风险操作结果: {result}")
    
    result = wrapped_safe()
    print(f"安全操作结果: {result}")
    
    # 4. 测试带参数的函数包装
    print("\n4. 测试带参数的函数包装:")
    def divide_numbers(a, b):
        if b == 0:
            raise ZeroDivisionError("除数不能为零")
        return a / b
    
    wrapped_divide = handler.wrap_error(
        func=divide_numbers,
        error_type=ValueError,
        message="除法操作错误",
        default_return=None
    )
    
    # 测试正常情况
    result = wrapped_divide(10, 2)
    print(f"正常除法结果: {result}")
    
    # 测试错误情况
    result = wrapped_divide(10, 0)
    print(f"错误除法结果: {result}")
    
    return handler

def test_input_validation():
    """
    测试输入验证功能
    """
    print("\n=== 测试输入验证功能 ===")
    
    handler = ErrorHandler()
    
    # 1. 测试有效输入
    print("\n1. 测试有效输入:")
    valid_inputs = [
        ({"key": "value"}, dict, "dict_input"),
        (123, int, "int_input"),
        ([1, 2, 3], list, "list_input"),
        ("string", str, "str_input"),
        (np.array([1, 2, 3]), np.ndarray, "array_input")
    ]
    
    for input_value, expected_type, param_name in valid_inputs:
        is_valid, error_msg = handler.validate_input(
            input_value=input_value,
            expected_type=expected_type,
            param_name=param_name
        )
        print(f"{param_name} 验证结果: {is_valid}, 错误消息: {error_msg}")
    
    # 2. 测试无效输入
    print("\n2. 测试无效输入:")
    invalid_inputs = [
        ("not a dict", dict, "dict_input"),
        ("not an int", int, "int_input"),
        (123, list, "list_input"),
        (123, str, "str_input"),
        ([1, 2, 3], np.ndarray, "array_input")
    ]
    
    for input_value, expected_type, param_name in invalid_inputs:
        is_valid, error_msg = handler.validate_input(
            input_value=input_value,
            expected_type=expected_type,
            param_name=param_name
        )
        print(f"{param_name} 验证结果: {is_valid}, 错误消息: {error_msg}")
    
    # 3. 测试带上下文的输入验证
    print("\n3. 测试带上下文的输入验证:")
    context = {"function": "process_image", "stage": "preprocessing"}
    is_valid, error_msg = handler.validate_input(
        input_value="not an array",
        expected_type=np.ndarray,
        param_name="image",
        context=context
    )
    print(f"带上下文验证结果: {is_valid}, 错误消息: {error_msg}")
    
    return handler

def test_custom_exceptions():
    """
    测试自定义异常类
    """
    print("\n=== 测试自定义异常类 ===")
    
    handler = ErrorHandler()
    
    # 1. 测试 ModelError
    print("\n1. 测试 ModelError:")
    try:
        error = FileNotFoundError("模型文件不存在")
        result = handler.handle_error(
            error=error,
            error_type=ModelError,
            message="模型加载失败",
            context={"model_path": "yolov8n.pt", "stage": "initialization"}
        )
        print(f"ModelError 处理结果: {result}")
    except Exception as e:
        print(f"ModelError 处理失败: {e}")
    
    # 2. 测试 ConfigurationError
    print("\n2. 测试 ConfigurationError:")
    try:
        error = TypeError("配置类型错误")
        result = handler.handle_error(
            error=error,
            error_type=ConfigurationError,
            message="配置无效",
            context={"config_key": "device", "config_value": "invalid", "expected_type": "str"}
        )
        print(f"ConfigurationError 处理结果: {result}")
    except Exception as e:
        print(f"ConfigurationError 处理失败: {e}")
    
    # 3. 测试 DataFormatError
    print("\n3. 测试 DataFormatError:")
    try:
        error = ValueError("数据格式错误")
        result = handler.handle_error(
            error=error,
            error_type=DataFormatError,
            message="输入数据格式无效",
            context={"expected_format": "RGB", "actual_format": "BGR", "input_shape": (480, 640, 3)}
        )
        print(f"DataFormatError 处理结果: {result}")
    except Exception as e:
        print(f"DataFormatError 处理失败: {e}")
    
    # 4. 测试 ProcessingError
    print("\n4. 测试 ProcessingError:")
    try:
        error = RuntimeError("处理失败")
        result = handler.handle_error(
            error=error,
            error_type=ProcessingError,
            message="图像处理失败",
            context={"processing_stage": "detection", "input_shape": (480, 640, 3)}
        )
        print(f"ProcessingError 处理结果: {result}")
    except Exception as e:
        print(f"ProcessingError 处理失败: {e}")
    
    return handler

def test_pipeline_error_handling():
    """
    测试管道中的错误处理
    """
    print("\n=== 测试管道中的错误处理 ===")
    
    handler = ErrorHandler()
    
    # 1. 测试无效配置
    print("\n1. 测试无效配置:")
    try:
        # 使用无效配置创建管道
        config = {
            "detector_config": {
                "model_path": "non_existent_model.pt",  # 不存在的模型文件
                "device": "cpu"
            }
        }
        
        pipeline = VisionPipeline(config)
        pipeline.initialize()
        print("管道初始化成功")
    except Exception as e:
        print(f"管道初始化失败 (预期行为): {e}")
    
    # 2. 测试有效配置但错误输入
    print("\n2. 测试有效配置但错误输入:")
    try:
        # 使用有效配置创建管道
        config = {
            "detector_config": {
                "model_path": "yolov8n.pt",
                "device": "cpu"
            }
        }
        
        pipeline = VisionPipeline(config)
        pipeline.initialize()
        print("管道初始化成功")
        
        # 测试错误输入
        print("测试错误输入:")
        # 传入非图像数据
        result = pipeline.process("not an image")
        print(f"处理结果: {result}")
    except Exception as e:
        print(f"处理失败 (预期行为): {e}")
    
    # 3. 测试批处理错误
    print("\n3. 测试批处理错误:")
    try:
        # 使用有效配置创建管道
        config = {
            "detector_config": {
                "model_path": "yolov8n.pt",
                "device": "cpu"
            }
        }
        
        pipeline = VisionPipeline(config)
        pipeline.initialize()
        print("管道初始化成功")
        
        # 创建有效和无效的混合输入
        valid_image = np.zeros((480, 640, 3), dtype=np.uint8)
        invalid_input = "not an image"
        
        batch = [valid_image, invalid_input, valid_image]
        results = pipeline.process_batch(batch)
        print(f"批处理结果数量: {len(results)}")
        
        for i, result in enumerate(results):
            if result is None:
                print(f"图像 {i+1} 处理失败 (预期行为)")
            else:
                print(f"图像 {i+1} 处理成功，检测到 {len(result.get('detections', []))} 个目标")
    except Exception as e:
        print(f"批处理失败: {e}")
    
    return handler

def test_error_handling_best_practices():
    """
    测试错误处理最佳实践
    """
    print("\n=== 测试错误处理最佳实践 ===")
    
    handler = ErrorHandler()
    
    # 1. 函数装饰器方式
    print("\n1. 函数装饰器方式:")
    
    def process_image(image):
        """
        处理图像的函数
        """
        # 验证输入
        is_valid, error_msg = handler.validate_input(
            input_value=image,
            expected_type=np.ndarray,
            param_name="image"
        )
        if not is_valid:
            return None, error_msg
        
        try:
            # 处理图像
            if len(image.shape) != 3:
                raise ValueError("图像必须是3通道")
            
            # 模拟处理
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return processed, None
        except Exception as e:
            # 处理错误
            result = handler.handle_error(
                error=e,
                error_type=ProcessingError,
                message="图像��理失败",
                context={"input_shape": image.shape if isinstance(image, np.ndarray) else "unknown"}
            )
            return None, str(result)
    
    # 测试函数
    print("测试处理函数:")
    
    # 测试有效输入
    valid_image = np.zeros((480, 640, 3), dtype=np.uint8)
    result, error = process_image(valid_image)
    print(f"有效输入处理结果: {'成功' if result is not None else '失败'}, 错误: {error}")
    
    # 测试无效输入
    invalid_input = "not an image"
    result, error = process_image(invalid_input)
    print(f"无效输入处理结果: {'成功' if result is not None else '失败'}, 错误: {error}")
    
    # 测试有效但格式错误的输入
    invalid_shape_image = np.zeros((480, 640), dtype=np.uint8)  # 单通道图像
    result, error = process_image(invalid_shape_image)
    print(f"格式错误输入处理结果: {'成功' if result is not None else '失败'}, 错误: {error}")
    
    # 2. 上下文管理器方式
    print("\n2. 上下文管理器方式:")
    
    class ErrorHandlingContext:
        """
        错误处理上下文管理器
        """
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if exc_val:
                handler = ErrorHandler()
                handler.handle_error(
                    error=exc_val,
                    error_type=VisionFrameworkError,
                    message="上下文管理器中的错误"
                )
                return True  # 抑制异常
    
    # 测试上下文管理器
    print("测试上下文管理器:")
    with ErrorHandlingContext():
        # 模拟错误
        raise ValueError("上下文管理器中的测试错误")
    print("上下文管理器执行完成 (错误已处理)")
    
    return handler

def main():
    """
    主函数
    """
    print("=== 错误处理示例 ===")
    
    try:
        # 测试基本错误处理功能
        basic_handler = test_basic_error_handling()
        
        # 测试错误包装功能
        wrapped_handler = test_error_wrapping()
        
        # 测试输入验证
        validation_handler = test_input_validation()
        
        # 测试自定义异常类
        custom_exceptions_handler = test_custom_exceptions()
        
        # 测试管道中的错误处理
        pipeline_handler = test_pipeline_error_handling()
        
        # 测试错误处理最佳实践
        best_practices_handler = test_error_handling_best_practices()
        
        print("\n=== 错误处理示例完成 ===")
        print("所有测试通过！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
