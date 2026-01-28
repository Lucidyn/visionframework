#!/usr/bin/env python3
"""
VisionPipeline 中使用 PyAV 示例

本示例展示如何在 VisionPipeline 中使用 PyAV 进行高性能视频处理，
包括基本用法、性能对比和实际应用场景。
"""

import os
import sys
import time
import numpy as np
import cv2

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionframework import VisionPipeline
from visionframework.utils.io import process_video


def create_test_video(output_path, duration=3, fps=30, width=640, height=480):
    """创建测试视频文件"""
    if os.path.exists(output_path):
        print(f"测试视频已存在: {output_path}")
        return
    
    print(f"创建测试视频: {output_path}")
    
    # 使用 OpenCV 创建测试视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(duration * fps):
        # 创建黑色背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加移动的矩形（模拟目标）
        square_size = 40
        x = int((width - square_size) * (i / (duration * fps)))
        y = int(height / 2 - square_size / 2)
        cv2.rectangle(frame, (x, y), (x + square_size, y + square_size), (0, 255, 0), -1)
        
        # 添加移动的圆形（模拟另一个目标）
        circle_radius = 20
        x2 = int((width - circle_radius * 2) * (1 - i / (duration * fps)))
        y2 = int(height / 2)
        cv2.circle(frame, (x2, y2), circle_radius, (0, 0, 255), -1)
        
        # 添加文本
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"测试视频创建完成: {output_path}")


def frame_callback(frame, frame_number, results):
    """帧回调函数：绘制检测和跟踪结果"""
    # 绘制检测结果
    for detection in results.get('detections', []):
        x1, y1, x2, y2 = detection['bbox']
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{detection['class_name']}: {detection['confidence']:.2f}", 
                    (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 绘制跟踪结果
    for track in results.get('tracks', []):
        x1, y1, x2, y2 = track['bbox']
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"ID: {track['track_id']}", 
                    (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame


def test_vision_pipeline_with_pyav():
    """测试 VisionPipeline 与 PyAV 集成"""
    print("\n=== VisionPipeline with PyAV 测试 ===")
    
    # 创建测试视频
    test_video = "test_pipeline.mp4"
    create_test_video(test_video)
    
    if not os.path.exists(test_video):
        print(f"错误: 测试视频不存在: {test_video}")
        return
    
    # 1. 测试基本的 VisionPipeline + PyAV
    print("\n1. 基本 VisionPipeline + PyAV 测试:")
    
    # 创建管道
    pipeline = VisionPipeline({
        "detector_config": {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.25
        },
        "enable_tracking": True,
        "tracker_config": {
            "tracker_type": "bytetrack"
        }
    })
    
    # 初始化管道
    pipeline.initialize()
    print("管道初始化成功")
    
    # 测试使用 OpenCV 处理（基准）
    print("\n   a. 使用 OpenCV 处理:")
    start_time = time.time()
    output_opencv = "output_opencv.mp4"
    success_opencv = pipeline.process_video(
        input_source=test_video,
        output_path=output_opencv,
        frame_callback=frame_callback,
        use_pyav=False
    )
    opencv_time = time.time() - start_time
    print(f"   处理时间: {opencv_time:.2f} 秒")
    print(f"   处理成功: {success_opencv}")
    
    # 测试使用 PyAV 处理
    print("\n   b. 使用 PyAV 处理:")
    start_time = time.time()
    output_pyav = "output_pyav.mp4"
    success_pyav = pipeline.process_video(
        input_source=test_video,
        output_path=output_pyav,
        frame_callback=frame_callback,
        use_pyav=True
    )
    pyav_time = time.time() - start_time
    print(f"   处理时间: {pyav_time:.2f} 秒")
    print(f"   处理成功: {success_pyav}")
    
    # 比较性能
    if opencv_time > 0:
        speedup = (opencv_time - pyav_time) / opencv_time * 100
        print(f"\n   性能比较:")
        print(f"   PyAV 比 OpenCV 快: {speedup:.1f}%")
    
    # 清理资源
    pipeline.cleanup()
    
    # 2. 测试简化 API
    print("\n2. 简化 API 测试:")
    
    output_simple = "output_simple.mp4"
    print("   使用 VisionPipeline.run_video() 静态方法:")
    start_time = time.time()
    success_simple = VisionPipeline.run_video(
        input_source=test_video,
        output_path=output_simple,
        model_path="yolov8n.pt",
        enable_tracking=True,
        use_pyav=True  # 启用 PyAV
    )
    simple_time = time.time() - start_time
    print(f"   处理时间: {simple_time:.2f} 秒")
    print(f"   处理成功: {success_simple}")
    
    # 清理输出文件
    print("\n3. 清理输出文件:")
    for file in [output_opencv, output_pyav, output_simple]:
        if os.path.exists(file):
            os.remove(file)
            print(f"   已清理: {file}")


def test_edge_cases():
    """测试边缘情况"""
    print("\n=== 边缘情况测试 ===")
    
    # 创建测试视频
    test_video = "test_edge_case.mp4"
    create_test_video(test_video, duration=1, fps=10, width=320, height=240)
    
    if not os.path.exists(test_video):
        print(f"错误: 测试视频不存在: {test_video}")
        return
    
    # 测试小视频
    print("\n1. 测试小视频处理:")
    
    pipeline = VisionPipeline({
        "detector_config": {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.25
        }
    })
    pipeline.initialize()
    
    output_small = "output_small.mp4"
    success = pipeline.process_video(
        input_source=test_video,
        output_path=output_small,
        use_pyav=True
    )
    print(f"   小视频处理成功: {success}")
    
    # 清理
    pipeline.cleanup()
    
    if os.path.exists(output_small):
        os.remove(output_small)
        print(f"   已清理: {output_small}")


def main():
    """主函数"""
    print("=== VisionPipeline with PyAV 示例 ===")
    
    try:
        # 测试 VisionPipeline 与 PyAV 集成
        test_vision_pipeline_with_pyav()
        
        # 测试边缘情况
        test_edge_cases()
        
        print("\n=== 示例完成 ===")
        print("\n总结:")
        print("1. VisionPipeline 现在支持通过 use_pyav 参数启用 PyAV 后端")
        print("2. PyAV 通常比 OpenCV 提供更高的视频处理性能")
        print("3. 当 PyAV 不可用时，系统会自动回退到 OpenCV")
        print("4. 简化 API VisionPipeline.run_video() 也支持 use_pyav 参数")
        
    finally:
        # 清理测试视频
        test_videos = ["test_video.mp4", "test_pipeline.mp4", "test_edge_case.mp4"]
        print("\n清理测试视频:")
        for video in test_videos:
            if os.path.exists(video):
                os.remove(video)
                print(f"   已清理: {video}")


if __name__ == "__main__":
    main()
