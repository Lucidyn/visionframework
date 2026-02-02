"""
03_video_processing.py
视频文件处理示例

这个示例展示了如何使用VisionFramework处理视频文件，包括：
1. 使用VisionPipeline处理本地视频文件
2. 使用run_video()静态方法一行代码处理视频
3. 处理结果可视化和保存

用法:
  python examples/03_video_processing.py
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
from visionframework.utils import ErrorHandler, is_dependency_available, validate_dependency


def create_test_video(output_path, duration=3.0, fps=30.0):
    """创建一个简单的测试视频"""
    print(f"创建测试视频: {output_path}...")
    
    width, height = 640, 480
    frames = int(duration * fps)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 定义移动对象
    objects = [
        {"name": "rectangle", "pos": [100, 100], "vel": [2, 1]},
        {"name": "circle", "pos": [320, 240], "vel": [-1, 2]}
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
        
        # 添加帧号
        cv2.putText(frame, f"Frame: {frame_idx+1}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 写入帧
        writer.write(frame)
    
    writer.release()
    print(f"测试视频已创建: {output_path}")
    return output_path


def process_video_with_pipeline(input_video, output_video):
    """使用VisionPipeline处理视频"""
    print(f"\n1. 使用VisionPipeline处理视频: {input_video}...")
    
    # 创建VisionPipeline实例，启用跟踪
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
    
    # 处理视频
    success = pipeline.process_video(
        input_source=input_video,
        output_path=output_video,
        frame_callback=frame_callback,
        progress_callback=progress_callback
    )
    
    if success:
        print(f"视频处理完成！结果保存到: {output_video}")
    else:
        print("视频处理失败！")
    
    return success


def process_video_with_simple_api(input_video, output_video):
    """使用简化API处理视频"""
    print(f"\n2. 使用简化API处理视频: {input_video}...")
    
    # 可视化回调函数
    viz = Visualizer()
    def frame_callback(frame, frame_number, results):
        detections = results.get("detections", [])
        tracks = results.get("tracks", [])
        
        # 绘制检测结果和跟踪结果
        vis_frame = viz.draw_detections(frame, detections)
        vis_frame = viz.draw_tracks(vis_frame, tracks, draw_history=True)
        
        return vis_frame
    
    # 一行代码处理视频
    success = VisionPipeline.run_video(
        input_source=input_video,
        output_path=output_video,
        model_path="yolov8n.pt",
        enable_tracking=True,
        conf_threshold=0.25,
        frame_callback=frame_callback
    )
    
    if success:
        print(f"视频处理完成！结果保存到: {output_video}")
    else:
        print("视频处理失败！")
    
    return success


def main():
    print("=== 视频文件处理示例 ===")
    
    # 1. 依赖检查
    print("\n1. 检查依赖...")
    handler = ErrorHandler()
    
    # 检查PyAV依赖（如果可用，将使用更高效的视频处理）
    pyav_available = is_dependency_available("pyav")
    print(f"PyAV依赖可用: {pyav_available}")
    if pyav_available:
        print("将使用PyAV进行更高效的视频处理")
    else:
        print("将使用OpenCV进行视频处理")
    
    try:
        # 测试视频路径
        test_video = "test_video.mp4"
        
        # 创建测试视频
        create_test_video(test_video)
        
        # 输出视频路径
        output_video1 = "output_video_pipeline.mp4"
        output_video2 = "output_video_simple.mp4"
        
        # 使用VisionPipeline处理视频
        process_video_with_pipeline(test_video, output_video1)
        
        # 使用简化API处理视频
        process_video_with_simple_api(test_video, output_video2)
        
        print("\n=== 示例完成 ===")
        print("视频处理示例展示了两种处理视频的方式：")
        print("  1. 使用VisionPipeline实例：适合需要精细控制的场景")
        print("  2. 使用run_video()静态方法：适合快速开发，一行代码完成任务")
        print("\n你可以替换test_video.mp4为自己的视频文件进行测试。")
        
    except Exception as e:
        # 统一错误处理
        handler.handle_error(
            error=e,
            error_type=RuntimeError,
            message="视频处理示例执行失败",
            context={"stage": "video_processing"},
            raise_error=False
        )
        print("示例执行过程中出现错误，但已被捕获和处理。")


if __name__ == "__main__":
    main()