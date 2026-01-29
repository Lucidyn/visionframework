"""
01_detection_with_tracking.py
带跟踪的目标检测示例

这个示例展示了如何使用VisionFramework进行带跟踪的目标检测，包括：
1. 创建带跟踪的VisionPipeline实例
2. 初始化检测器和跟踪器
3. 处理图像序列
4. 可视化检测和跟踪结果

用法:
  python examples/01_detection_with_tracking.py
"""
import os
import cv2
import numpy as np
from pathlib import Path

# 修复OpenMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from visionframework import VisionPipeline, Visualizer


def main():
    print("=== 带跟踪的目标检测示例 ===")
    
    # 创建一个简单的动画场景作为测试
    print("创建测试动画场景...")
    
    # 图像尺寸
    width, height = 640, 480
    
    # 创建VisionPipeline实例，启用跟踪
    print("创建带跟踪的VisionPipeline实例...")
    config = {
        "enable_tracking": True,  # 启用跟踪
        "detector_config": {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.25
        },
        "tracker_config": {
            "tracker_type": "bytetrack",  # 使用ByteTrack跟踪器
            "max_age": 30  # 最大跟踪年龄
        }
    }
    
    pipeline = VisionPipeline(config)
    
    print("初始化检测器和跟踪器...")
    if not pipeline.initialize():
        print("初始化失败！")
        return
    
    print("开始处理图像序列...")
    
    # 创建可视化器
    viz = Visualizer()
    
    # 定义一个简单的动画：三个形状在图像中移动
    # 初始位置和速度
    objects = [
        {"name": "rectangle", "pos": [100, 100], "vel": [2, 1]},
        {"name": "circle", "pos": [320, 240], "vel": [-1, 2]},
        {"name": "triangle", "pos": [500, 300], "vel": [1, -1]}
    ]
    
    # 创建视频编写器，保存动画结果
    output_video = "output_tracking_demo.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, 30.0, (width, height))
    
    # 处理300帧动画
    for frame_idx in range(300):
        # 创建空白帧
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (128, 128, 128)  # 灰色背景
        
        # 更新对象位置并绘制
        for obj in objects:
            # 更新位置
            obj["pos"][0] += obj["vel"][0]
            obj["pos"][1] += obj["vel"][1]
            
            # 边界检测和反弹
            if obj["pos"][0] <= 50 or obj["pos"][0] >= width - 50:
                obj["vel"][0] *= -1
            if obj["pos"][1] <= 50 or obj["pos"][1] >= height - 50:
                obj["vel"][1] *= -1
            
            # 绘制对象
            x, y = obj["pos"]
            if obj["name"] == "rectangle":
                cv2.rectangle(frame, (x-30, y-30), (x+30, y+30), (255, 0, 0), -1)
            elif obj["name"] == "circle":
                cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)
            elif obj["name"] == "triangle":
                pts = np.array([[x, y-30], [x+30, y+30], [x-30, y+30]], np.int32)
                cv2.fillPoly(frame, [pts], (0, 0, 255))
        
        # 处理当前帧
        results = pipeline.process(frame)
        detections = results.get("detections", [])
        tracks = results.get("tracks", [])
        
        # 可视化结果
        vis_frame = viz.draw_detections(frame, detections)
        vis_frame = viz.draw_tracks(vis_frame, tracks, draw_history=True)
        
        # 添加帧率信息
        cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Detections: {len(detections)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Tracks: {len(tracks)}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 保存当前帧到视频
        writer.write(vis_frame)
        
        # 每30帧显示一次进度
        if frame_idx % 30 == 0:
            print(f"处理第 {frame_idx} 帧，检测到 {len(detections)} 个目标，跟踪到 {len(tracks)} 个目标")
    
    # 释放视频编写器
    writer.release()
    print(f"\n动画处理完成！结果已保存到: {output_video}")
    
    # 保存最后一帧的可视化结果
    cv2.imwrite("output_tracking_final.jpg", vis_frame)
    print(f"最后一帧结果已保存到: output_tracking_final.jpg")
    
    print("\n=== 示例完成 ===")
    print("你可以使用视频播放器查看生成的跟踪演示视频。")


if __name__ == "__main__":
    main()