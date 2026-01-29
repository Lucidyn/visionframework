#!/usr/bin/env python3
"""
PyAV 视频处理示例

本示例展示如何使用 PyAV 进行高性能视频处理，
并与 OpenCV 的视频处理性能进行比较。

PyAV 是基于 FFmpeg 的视频处理库，通常比 OpenCV 的视频处理性能更高，
特别是在处理高分辨率视频时。
"""

import os
import sys
import time
import numpy as np
import cv2

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visionframework.utils.io import process_video, PyAVVideoProcessor, PyAVVideoWriter


# 创建测试视频（如果不存在）
def create_test_video(output_path, duration=5, fps=30, width=1280, height=720):
    """创建测试视频文件"""
    if os.path.exists(output_path):
        print(f"测试视频已存在: {output_path}")
        return
    
    print(f"创建测试视频: {output_path}")
    
    # 使用 OpenCV 创建测试视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i in range(duration * fps):
        # 创建渐变背景
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 添加移动的矩形
        x = int((i / (duration * fps)) * (width - 200))
        cv2.rectangle(frame, (x, height//2 - 50), (x + 200, height//2 + 50), (0, 255, 0), -1)
        
        # 添加文本
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"测试视频创建完成: {output_path}")


# 简单的帧处理函数
def simple_process_frame(frame, frame_number):
    """简单的帧处理函数：转换为灰度并返回"""
    # 转换为灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 转换回 BGR
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# 性能测试函数
def benchmark_video_processing(video_path):
    """比较 OpenCV 和 PyAV 的视频处理性能"""
    print("\n=== 视频处理性能测试 ===")
    print(f"测试视频: {video_path}")
    
    # 使用 OpenCV 处理
    print("\n1. 使用 OpenCV 处理:")
    start_time = time.time()
    success = process_video(
        input_path=video_path,
        output_path="opencv_output.mp4",
        frame_callback=simple_process_frame,
        use_pyav=False
    )
    opencv_time = time.time() - start_time
    print(f"   处理时间: {opencv_time:.2f} 秒")
    print(f"   处理成功: {success}")
    
    # 使用 PyAV 处理
    print("\n2. 使用 PyAV 处理:")
    start_time = time.time()
    success = process_video(
        input_path=video_path,
        output_path="pyav_output.mp4",
        frame_callback=simple_process_frame,
        use_pyav=True
    )
    pyav_time = time.time() - start_time
    print(f"   处理时间: {pyav_time:.2f} 秒")
    print(f"   处理成功: {success}")
    
    # 计算性能提升
    if opencv_time > 0:
        speedup = (opencv_time - pyav_time) / opencv_time * 100
        print(f"\n3. 性能比较:")
        print(f"   PyAV 比 OpenCV 快: {speedup:.1f}%")
        
        if speedup > 0:
            print("   ✓ PyAV 表现更好")
        else:
            print("   ✓ OpenCV 表现更好")


# 直接使用 PyAVVideoProcessor 的示例
def direct_pyav_example(video_path):
    """直接使用 PyAVVideoProcessor 进行视频处理"""
    print("\n=== 直接使用 PyAVVideoProcessor 示例 ===")
    
    try:
        from visionframework.utils.io import PyAVVideoProcessor
        
        # 使用上下文管理器
        with PyAVVideoProcessor(video_path) as processor:
            info = processor.get_info()
            print(f"视频信息: {info}")
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = processor.read_frame()
                if not ret:
                    break
                
                # 简单处理：添加文本
                cv2.putText(frame, f"Processed with PyAV", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                frame_count += 1
                
                # 每10帧显示一次
                if frame_count % 10 == 0:
                    print(f"处理帧: {frame_count}")
            
            elapsed_time = time.time() - start_time
            print(f"\n处理完成:")
            print(f"总帧数: {frame_count}")
            print(f"处理时间: {elapsed_time:.2f} 秒")
            print(f"平均帧率: {frame_count / elapsed_time:.2f} FPS")
            
    except ImportError as e:
        print(f"PyAV 未安装: {e}")
    except Exception as e:
        print(f"错误: {e}")


# RTSP流处理示例
def rtsp_stream_example():
    """使用 PyAV 处理 RTSP 流"""
    print("\n=== RTSP 流处理示例 ===")
    
    # 示例 RTSP URL（请替换为实际的 RTSP 流地址）
    rtsp_url = "rtsp://example.com/stream"
    
    print(f"尝试使用 PyAV 处理 RTSP 流: {rtsp_url}")
    print("注意：请将上面的 URL 替换为实际的 RTSP 流地址")
    print("例如：rtsp://username:password@192.168.1.100:554/stream1")
    
    try:
        from visionframework.utils.io import PyAVVideoProcessor
        
        # 使用上下文管理器处理 RTSP 流
        with PyAVVideoProcessor(rtsp_url) as processor:
            info = processor.get_info()
            print(f"流信息: {info}")
            
            frame_count = 0
            start_time = time.time()
            max_frames = 30  # 只处理30帧以避免无限循环
            
            print(f"开始处理 RTSP 流，将处理 {max_frames} 帧...")
            
            while frame_count < max_frames:
                ret, frame = processor.read_frame()
                if not ret:
                    print("无法读取帧，退出")
                    break
                
                # 简单处理：添加文本
                cv2.putText(frame, f"Processed with PyAV (Frame {frame_count})", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                frame_count += 1
                
                # 每5帧显示一次
                if frame_count % 5 == 0:
                    print(f"处理帧: {frame_count}")
            
            elapsed_time = time.time() - start_time
            print(f"\n处理完成:")
            print(f"处理帧数: {frame_count}")
            print(f"处理时间: {elapsed_time:.2f} 秒")
            if frame_count > 0:
                print(f"平均帧率: {frame_count / elapsed_time:.2f} FPS")
                
    except ImportError as e:
        print(f"PyAV 未安装: {e}")
    except Exception as e:
        print(f"错误: {e}")
        print("请确保 RTSP 流地址正确且可访问")

# 主函数
def main():
    """主函数"""
    print("=== PyAV 视频处理示例 ===")
    
    # 创建测试视频
    test_video_path = "test_video.mp4"
    create_test_video(test_video_path)
    
    # 检查文件是否存在
    if not os.path.exists(test_video_path):
        print(f"错误: 测试视频不存在: {test_video_path}")
        return
    
    # 运行性能测试
    benchmark_video_processing(test_video_path)
    
    # 运行直接使用 PyAV 的示例
    direct_pyav_example(test_video_path)
    
    # 运行 RTSP 流处理示例
    rtsp_stream_example()
    
    # 清理输出文件
    for file in ["opencv_output.mp4", "pyav_output.mp4"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"已清理输出文件: {file}")
    
    print("\n示例完成！")


if __name__ == "__main__":
    main()
