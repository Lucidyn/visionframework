"""
05_advanced_features.py
高级功能示例

这个示例展示了VisionFramework的一些高级功能，包括：
1. 使用配置文件配置管道
2. 模型管理和下载
3. 批量图像处理
4. 结果导出

用法:
  python examples/05_advanced_features.py
"""
import os
import cv2
import numpy as np
import json
from pathlib import Path

# 修复OpenMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from visionframework import VisionPipeline, Visualizer, ModelManager
from visionframework.utils import ResultExporter


def create_test_images(count=3):
    """创建测试图像列表"""
    print(f"创建 {count} 个测试图像...")
    images = []
    
    for i in range(count):
        # 创建空白图像
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (128, 128, 128)  # 灰色背景
        
        # 绘制不同的形状
        if i == 0:
            # 图像1：绘制矩形
            cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), -1)
        elif i == 1:
            # 图像2：绘制圆形
            cv2.circle(frame, (320, 240), 50, (0, 255, 0), -1)
        else:
            # 图像3：绘制三角形
            pts = np.array([[500, 100], [550, 200], [450, 200]], np.int32)
            cv2.fillPoly(frame, [pts], (0, 0, 255))
        
        images.append(frame)
    
    print(f"已创建 {len(images)} 个测试图像")
    return images


def test_model_manager():
    """测试模型管理功能"""
    print("\n1. 测试模型管理功能...")
    
    # 获取模型管理器实例
    model_manager = ModelManager()
    
    # 列出注册的模型
    print("   注册的模型列表：")
    registered_models = model_manager.get_all_registered_models()
    for model in registered_models[:5]:  # 只显示前5个
        print(f"   - {model}")
    
    # 检查特定模型是否已缓存
    model_name = "yolov8n"
    # 直接获取模型路径（如果不存在会自动下载）
    try:
        model_path = model_manager.get_model_path(model_name)
        print(f"   模型 {model_name} 已可用，路径：{model_path}")
    except Exception as e:
        print(f"   获取模型 {model_name} 失败：{e}")


def test_batch_processing():
    """测试批量图像处理"""
    print("\n2. 测试批量图像处理...")
    
    # 创建测试图像
    images = create_test_images(3)
    
    # 创建VisionPipeline实例
    config = {
        "enable_tracking": False,
        "detector_config": {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.25,
            "batch_inference": True  # 启用批量推理
        }
    }
    
    pipeline = VisionPipeline(config)
    if not pipeline.initialize():
        print("   初始化失败！")
        return False
    
    # 批量处理图像
    print("   开始批量处理图像...")
    results_list = pipeline.process_batch(images)
    
    # 处理结果
    for i, results in enumerate(results_list):
        detections = results.get("detections", [])
        print(f"   图像 {i+1} 检测到 {len(detections)} 个目标")
    
    return True


def test_config_file():
    """测试配置文件功能"""
    print("\n3. 测试配置文件功能...")
    
    # 创建一个临时配置文件
    config_data = {
        "enable_tracking": True,
        "detector_config": {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.25
        },
        "tracker_config": {
            "tracker_type": "bytetrack",
            "max_age": 30
        }
    }
    
    # 保存配置文件
    config_file = "temp_config.json"
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"   已创建配置文件: {config_file}")
    
    # 使用配置文件创建管道
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    pipeline = VisionPipeline(config)
    if pipeline.initialize():
        print("   使用配置文件成功初始化管道")
    else:
        print("   使用配置文件初始化管道失败")
        return False
    
    # 删除临时配置文件
    os.remove(config_file)
    print(f"   已删除临时配置文件: {config_file}")
    
    return True


def test_result_export():
    """测试结果导出功能"""
    print("\n4. 测试结果导出功能...")
    
    # 创建测试图像
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (128, 128, 128)
    cv2.rectangle(image, (100, 100), (200, 200), (255, 0, 0), -1)
    
    # 创建并初始化管道
    pipeline = VisionPipeline({
        "enable_tracking": True,
        "detector_config": {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.25
        }
    })
    
    if not pipeline.initialize():
        print("   初始化失败！")
        return False
    
    # 处理图像
    results = pipeline.process(image)
    detections = results.get("detections", [])
    tracks = results.get("tracks", [])
    
    print(f"   检测到 {len(detections)} 个目标，{len(tracks)} 个跟踪")
    
    # 导出结果
    exporter = ResultExporter()
    
    # 导出为JSON
    json_output = "detections_results.json"
    exporter.export_detections_to_json(detections, json_output)
    print(f"   结果已导出到JSON文件: {json_output}")
    
    return True


def main():
    print("=== 高级功能示例 ===")
    
    # 测试模型管理
    test_model_manager()
    
    # 测试批量处理
    test_batch_processing()
    
    # 测试配置文件
    test_config_file()
    
    # 测试结果导出
    test_result_export()
    
    print("\n=== 示例完成 ===")
    print("高级功能示例展示了VisionFramework的多种高级特性：")
    print("  1. 模型管理：下载、检查和管理模型")
    print("  2. 批量处理：高效处理多个图像")
    print("  3. 配置文件：使用JSON文件配置管道")
    print("  4. 结果导出：将检测结果导出为JSON格式")
    print("\n这些高级功能可以帮助你更好地使用VisionFramework进行复杂的计算机视觉任务。")


if __name__ == "__main__":
    main()