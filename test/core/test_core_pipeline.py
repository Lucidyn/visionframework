"""
测试核心管道功能
"""

import pytest
import numpy as np
from visionframework.core.pipelines import VisionPipeline


def test_vision_pipeline_initialization():
    """测试视觉管道初始化"""
    # 测试默认初始化
    pipeline = VisionPipeline()
    assert pipeline is not None
    
    # 测试使用配置初始化
    config = {
        "detector_config": {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.7,
            "device": "cpu"
        },
        "tracker_config": {
            "tracker_type": "bytetrack",
            "max_age": 30
        }
    }
    
    pipeline = VisionPipeline(config)
    assert pipeline is not None


def test_vision_pipeline_with_pose_estimation():
    """测试带姿态估计的视觉管道"""
    config = {
        "enable_pose_estimation": True,
        "detector_config": {
            "model_path": "yolov8n.pt",
            "device": "cpu"
        },
        "pose_estimator_config": {
            "model_path": "yolov8n-pose.pt",
            "device": "cpu"
        }
    }
    
    pipeline = VisionPipeline(config)
    assert pipeline is not None
    assert pipeline.config["enable_pose_estimation"] == True


def test_vision_pipeline_batch_processing():
    """测试视觉管道批处理功能"""
    config = {
        "detector_config": {
            "model_path": "yolov8n.pt",
            "device": "cpu"
        }
    }
    
    pipeline = VisionPipeline(config)
    assert pipeline is not None
    
    # 创建测试图像
    test_images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
    
    # 只测试管道初始化和配置，不实际运行批处理（避免加载模型）
    assert pipeline.config is not None
    assert "detector_config" in pipeline.config
    assert pipeline.config["detector_config"]["model_path"] == "yolov8n.pt"
    assert pipeline.config["detector_config"]["device"] == "cpu"


def test_vision_pipeline_batch_processing_parameters():
    """测试视觉管道批处理参数"""
    config = {
        "detector_config": {
            "model_path": "yolov8n.pt",
            "device": "cpu"
        }
    }
    
    pipeline = VisionPipeline(config)
    assert pipeline is not None
    
    # 创建测试图像
    test_images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
    
    # 测试不同参数组合（不实际运行，只测试参数接受）
    # 这里我们只测试方法签名和参数接受，不实际执行批处理
    # 因为实际执行需要加载模型，会增加测试时间和资源消耗
    
    # 测试默认参数
    try:
        # 注意：这里我们不实际调用process_batch，因为会加载模型
        # 我们只是验证方法存在并且可以接受参数
        assert hasattr(pipeline, 'process_batch')
        # 检查方法签名
        import inspect
        sig = inspect.signature(pipeline.process_batch)
        params = list(sig.parameters.keys())
        assert 'images' in params
        assert 'max_batch_size' in params
        assert 'use_parallel' in params
        assert 'max_workers' in params
        assert 'enable_memory_optimization' in params
    except Exception as e:
        assert False, f"Error testing process_batch parameters: {e}"


def test_vision_pipeline_process_video_batch():
    """测试视觉管道视频批处理功能"""
    config = {
        "detector_config": {
            "model_path": "yolov8n.pt",
            "device": "cpu"
        }
    }
    
    pipeline = VisionPipeline(config)
    assert pipeline is not None
    
    # 测试方法存在
    assert hasattr(pipeline, 'process_video_batch')
    
    # 检查方法签名
    import inspect
    sig = inspect.signature(pipeline.process_video_batch)
    params = list(sig.parameters.keys())
    assert 'input_source' in params
    assert 'output_path' in params
    assert 'use_pyav' in params
    
    # 验证use_pyav参数存在
    assert 'use_pyav' in params


def test_vision_pipeline_get_config():
    """测试获取管道配置"""
    config = {
        "detector_config": {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.7,
            "device": "cpu"
        }
    }
    
    pipeline = VisionPipeline(config)
    pipeline_config = pipeline.config
    
    assert isinstance(pipeline_config, dict)
    assert "detector_config" in pipeline_config
    assert "tracker_config" in pipeline_config
    assert "pose_estimator_config" in pipeline_config
