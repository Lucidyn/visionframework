"""
04_stream_processing.py
视频流处理示例

这个示例展示了如何使用VisionFramework处理视频流，包括：
1. 处理RTSP流
2. 处理HTTP流
3. 使用简化API处理视频流

用法:
  python examples/04_stream_processing.py
  python examples/04_stream_processing.py --stream rtsp://example.com/stream
"""
import os
import cv2
import numpy as np
import argparse
from pathlib import Path

# 修复OpenMP库冲突
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from visionframework import VisionPipeline, Visualizer


def create_stream_simulation(output_video, duration=5.0, fps=30.0):
    """创建一个模拟流的视频文件"""
    print(f"创建流模拟视频: {output_video}...")
    
    width, height = 640, 480
    frames = int(duration * fps)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 定义移动对象
    objects = [
        {"name": "rectangle", "pos": [100, 100], "vel": [2, 1]},
        {"name": "circle", "pos": [320, 240], "vel": [-1, 2]},
        {"name": "triangle", "pos": [500, 300], "vel": [1, -1]}
    ]
    
    for frame_idx in range(frames):
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
        
        # 添加流模拟信息
        cv2.putText(frame, "Stream Simulation", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx+1}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 写入帧
        writer.write(frame)
    
    writer.release()
    print(f"流模拟视频已创建: {output_video}")
    return output_video


def process_stream(stream_url, output_path=None, use_simple_api=False):
    """处理视频流"""
    method_name = "简化API" if use_simple_api else "VisionPipeline实例"
    print(f"\n使用{method_name}处理流: {stream_url}")
    
    # 可视化回调函数
    viz = Visualizer()
    def frame_callback(frame, frame_number, results):
        detections = results.get("detections", [])
        tracks = results.get("tracks", [])
        
        # 绘制检测结果和跟踪结果
        vis_frame = viz.draw_detections(frame, detections)
        vis_frame = viz.draw_tracks(vis_frame, tracks, draw_history=True)
        
        return vis_frame
    
    # 进度回调函数
    def progress_callback(progress, current_frame, total_frames):
        print(f"   进度: {progress:.1%} ({current_frame}/{total_frames})")
    
    if use_simple_api:
        # 使用简化API处理流
        success = VisionPipeline.run_video(
            input_source=stream_url,
            output_path=output_path,
            model_path="yolov8n.pt",
            enable_tracking=True,
            conf_threshold=0.25,
            start_frame=0,
            end_frame=100,  # 只处理100帧用于演示
            frame_callback=frame_callback,
            progress_callback=progress_callback
        )
    else:
        # 使用VisionPipeline实例处理流
        config = {
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
        
        pipeline = VisionPipeline(config)
        if not pipeline.initialize():
            print("初始化失败！")
            return False
        
        success = pipeline.process_video(
            input_source=stream_url,
            output_path=output_path,
            start_frame=0,
            end_frame=100,  # 只处理100帧用于演示
            frame_callback=frame_callback,
            progress_callback=progress_callback
        )
    
    if success:
        print(f"流处理完成！{'结果保存到: ' + output_path if output_path else ''}")
    else:
        print("流处理失败！")
    
    return success


def main():
    parser = argparse.ArgumentParser(description="视频流处理示例")
    parser.add_argument('--stream', type=str, default=None, 
                      help='RTSP/HTTP流URL或视频文件路径')
    parser.add_argument('--output', type=str, default=None, 
                      help='输出视频文件路径')
    args = parser.parse_args()
    
    print("=== 视频流处理示例 ===")
    
    # 流URL或视频文件
    stream_url = args.stream
    
    # 如果没有提供流URL，创建一个模拟流的视频文件
    if stream_url is None:
        print("未提供流URL，创建流模拟视频...")
        stream_url = create_stream_simulation("stream_simulation.mp4")
        print(f"使用模拟流: {stream_url}")
    
    # 输出路径
    output_path1 = args.output if args.output else "output_stream_pipeline.mp4"
    output_path2 = "output_stream_simple.mp4"
    
    print("\n=== 流处理示例 ===")
    print("注意：真实流处理可能需要较长时间，示例中只处理前100帧")
    
    # 1. 使用VisionPipeline实例处理流
    process_stream(stream_url, output_path1, use_simple_api=False)
    
    # 2. 使用简化API处理流
    process_stream(stream_url, output_path2, use_simple_api=True)
    
    print("\n=== 示例完成 ===")
    print("视频流处理示例展示了两种处理流的方式：")
    print("  1. 使用VisionPipeline实例：适合需要精细控制的场景")
    print("  2. 使用run_video()静态方法：适合快速开发，一行代码完成任务")
    print("\n你可以使用真实的RTSP/HTTP流URL进行测试，例如：")
    print("  python examples/04_stream_processing.py --stream rtsp://example.com/stream")
    print("  python examples/04_stream_processing.py --stream http://example.com/stream.mjpg")


if __name__ == "__main__":
    main()