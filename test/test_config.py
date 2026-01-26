"""
测试配置文件加载和管道初始化
"""

import json
from visionframework.core.pipeline import VisionPipeline


def test_config_loading():
    """
    测试配置文件加载
    """
    print("=== 测试配置文件加载 ===")
    
    # 加载配置文件
    config_path = "examples/my_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(f"✓ 配置文件加载成功: {config_path}")
    
    # 打印配置信息
    print("\n=== 配置信息 ===")
    print(f"  检测器配置: {config.get('detector', {}).get('model_path', 'N/A')}")
    print(f"  跟踪器配置: 启用={config.get('pipeline', {}).get('enable_tracking', False)}")
    print(f"  姿态估计: 启用={config.get('pipeline', {}).get('enable_pose_estimation', False)}")
    print(f"  CLIP特征: 启用={config.get('pipeline', {}).get('enable_clip', False)}")
    print(f"  SAM分割: 启用={config.get('pipeline', {}).get('enable_sam', False)}")
    
    return config


def test_pipeline_initialization(config):
    """
    测试管道初始化
    """
    print("\n=== 测试管道初始化 ===")
    
    try:
        pipeline_config = config.get("pipeline", {})
        pipeline = VisionPipeline(pipeline_config)
        print("✓ 管道对象创建成功")
        
        # 检查属性
        print(f"  enable_tracking: {pipeline.enable_tracking}")
        print(f"  enable_pose_estimation: {pipeline.enable_pose_estimation}")
        print(f"  enable_performance_monitoring: {pipeline.enable_performance_monitoring}")
        print(f"  performance_metrics: {pipeline.performance_metrics}")
        
        return True
    except Exception as e:
        print(f"✗ 管道初始化失败: {e}")
        return False


if __name__ == "__main__":
    config = test_config_loading()
    test_pipeline_initialization(config)
    print("\n=== 测试完成 ===")
