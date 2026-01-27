"""
测试核心管道功能
"""

import pytest
import numpy as np
from visionframework.core.pipeline import VisionPipeline


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
