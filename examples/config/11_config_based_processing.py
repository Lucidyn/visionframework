"""
配置文件驱动的视觉处理示例

本示例展示如何使用配置文件来初始化和运行视觉处理管道，
实现对象检测、跟踪、姿态估计、CLIP特征提取和SAM分割。
"""

import json
import cv2
import numpy as np
from visionframework import VisionPipeline, Visualizer


def load_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path (str): 配置文件路径
    
    Returns:
        dict: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def create_demo_image():
    """
    创建演示图像
    
    Returns:
        np.ndarray: 演示图像
    """
    # 创建一个简单的演示图像
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (240, 240, 240)  # 浅灰色背景
    
    # 绘制一个人
    cv2.circle(img, (320, 150), 30, (0, 0, 255), -1)  # 头部
    cv2.rectangle(img, (300, 180), (340, 280), (0, 0, 255), -1)  # 身体
    cv2.line(img, (320, 180), (280, 240), (0, 0, 255), 5)  # 左臂
    cv2.line(img, (320, 180), (360, 240), (0, 0, 255), 5)  # 右臂
    cv2.line(img, (320, 280), (300, 380), (0, 0, 255), 5)  # 左腿
    cv2.line(img, (320, 280), (340, 380), (0, 0, 255), 5)  # 右腿
    
    # 绘制一辆车
    cv2.rectangle(img, (150, 350), (250, 400), (255, 0, 0), -1)  # 车身
    cv2.circle(img, (170, 400), 10, (0, 0, 0), -1)  # 左车轮
    cv2.circle(img, (230, 400), 10, (0, 0, 0), -1)  # 右车轮
    
    return img


def main():
    """
    主函数
    """
    print("=== 配置文件驱动的视觉处理示例 ===")
    
    # 加载配置文件
    config_path = "examples/config/my_config.json"
    config = load_config(config_path)
    print(f"✓ 加载配置文件: {config_path}")
    
    # 初始化视觉管道
    pipeline_config = config.get("pipeline", {})
    print("✓ 初始化视觉管道...")
    
    try:
        pipeline = VisionPipeline(pipeline_config)
        pipeline.initialize()
        print("✓ 视觉管道初始化成功")
        
        # 显示配置信息
        print("\n=== 配置信息 ===")
        print(f"  目标检测: {'启用' if True else '禁用'}")
        print(f"  目标跟踪: {'启用' if pipeline_config.get('enable_tracking', False) else '禁用'}")
        print(f"  姿态估计: {'启用' if pipeline_config.get('enable_pose_estimation', False) else '禁用'}")
        print(f"  CLIP特征: {'启用' if pipeline_config.get('enable_clip', False) else '禁用'}")
        print(f"  SAM分割: {'启用' if pipeline_config.get('enable_sam', False) else '禁用'}")
        
        # 创建演示图像
        frame = create_demo_image()
        print("✓ 创建演示图像")
        
        # 处理图像
        print("\n=== 处理图像 ===")
        results = pipeline.process(frame)
        print("✓ 图像处理完成")
        
        # 显示结果
        print("\n=== 处理结果 ===")
        if results.get("detections"):
            detections = results["detections"]
            print(f"  检测到 {len(detections)} 个目标")
            for i, detection in enumerate(detections):
                print(f"    {i+1}. {detection.class_name}: {detection.confidence:.2f}")
        else:
            print("  未检测到目标")
        
        # 可视化结果
        visualizer = Visualizer()
        detections = results.get("detections", [])
        tracks = results.get("tracks", [])
        poses = results.get("poses", [])
        vis_frame = visualizer.draw_results(frame, detections, tracks, poses)
        
        # 保存结果图像
        output_path = "config_based_processing_result.jpg"
        cv2.imwrite(output_path, vis_frame)
        print("\n=== 显示结果 ===")
        print(f"  结果已保存到: {output_path}")
        print("  处理完成！")
        
    except Exception as e:
        print(f"✗ 错误: {e}")
    finally:
        if 'pipeline' in locals():
            pipeline.shutdown()


if __name__ == "__main__":
    main()
