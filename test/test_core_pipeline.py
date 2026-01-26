"""
测试核心管道功能
"""

import pytest
from visionframework.core.pipeline import VisionPipeline


def test_vision_pipeline_initialization():
    """测试视觉管道初始化"""
    # 测试默认初始化
    pipeline = VisionPipeline()
    assert pipeline is not None
    
    # 测试使用配置初始化
    config = {
        "detector": {
            "model_type": "yolo",
            "conf_threshold": 0.7,
            "device": "cpu"
        },
        "tracker": {
            "tracker_type": "iou",
            "iou_threshold": 0.5
        }
    }
    
    pipeline = VisionPipeline(config)
    assert pipeline is not None


def test_vision_pipeline_get_config():
    """测试获取管道配置"""
    config = {
        "detector": {
            "model_type": "yolo",
            "conf_threshold": 0.7,
            "device": "cpu"
        }
    }
    
    pipeline = VisionPipeline(config)
    pipeline_config = pipeline.get_config()
    
    assert isinstance(pipeline_config, dict)
    assert "detector" in pipeline_config
    assert "detector_config" in pipeline_config
    assert "tracker_config" in pipeline_config
