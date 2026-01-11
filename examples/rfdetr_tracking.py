"""
RF-DETR 检测 + 跟踪示例

本示例展示如何使用 RF-DETR 检测器进行目标检测和跟踪。
RF-DETR 是 Roboflow 开发的高性能实时目标检测模型，结合跟踪器可以实现稳定的多目标跟踪。

本示例包含：
- RF-DETR + IoU 跟踪器
- RF-DETR + ByteTrack 跟踪器
- 视频文件处理
"""

import cv2
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from visionframework import Detector, Tracker, VisionPipeline, Visualizer


def example_rfdetr_with_iou_tracker():
    """
    示例 1: RF-DETR 检测 + IoU 跟踪器
    
    本示例展示如何使用 RF-DETR 检测器配合 IoU 跟踪器进行目标跟踪。
    IoU 跟踪器基于 IoU（交并比）匹配，适合简单场景。
    """
    print("=" * 70)
    print("示例 1: RF-DETR 检测 + IoU 跟踪器")
    print("=" * 70)
    
    # ========== 步骤 1: 初始化 RF-DETR 检测器 ==========
    print("\n1. 初始化 RF-DETR 检测器...")
    
    detector = Detector({
        "model_type": "rfdetr",
        "conf_threshold": 0.5,
        "device": "cpu"  # 可选: "cpu", "cuda", "mps"
    })
    
    if not detector.initialize():
        print("✗ 检测器初始化失败！")
        print("  提示: 请确保已安装 rfdetr: pip install rfdetr supervision")
        return
    
    print("  ✓ RF-DETR 检测器初始化成功")
    
    # ========== 步骤 2: 初始化跟踪器 ==========
    print("\n2. 初始化 IoU 跟踪器...")
    
    tracker = Tracker({
        "tracker_type": "iou",      # 使用 IoU 跟踪器
        "max_age": 30,              # 目标丢失最大帧数
        "min_hits": 3,              # 确认跟踪的最小命中次数
        "iou_threshold": 0.3        # IoU 匹配阈值
    })
    
    if not tracker.initialize():
        print("✗ 跟踪器初始化失败！")
        return
    
    print("  ✓ IoU 跟踪器初始化成功")
    
    # ========== 步骤 3: 初始化可视化器 ==========
    visualizer = Visualizer({
        "show_labels": True,
        "show_confidences": True,
        "show_track_ids": True,
        "line_thickness": 2
    })
    
    # ========== 步骤 4: 创建测试视频序列 ==========
    print("\n3. 创建测试视频序列...")
    
    # 创建多帧测试图像，模拟对象移动
    frames = []
    h, w = 640, 480
    
    for i in range(30):  # 创建30帧
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (50, 50, 50)  # 深灰色背景
        
        # 模拟移动的矩形对象
        x_offset = int(50 + i * 5)
        y_offset = int(100 + i * 3)
        cv2.rectangle(frame, 
                     (x_offset, y_offset), 
                     (x_offset + 100, y_offset + 100), 
                     (0, 255, 0), -1)
        
        # 添加另一个对象
        x2_offset = int(300 - i * 3)
        y2_offset = int(200 + i * 2)
        cv2.rectangle(frame,
                     (x2_offset, y2_offset),
                     (x2_offset + 80, y2_offset + 80),
                     (255, 0, 0), -1)
        
        frames.append(frame)
    
    print(f"  ✓ 创建了 {len(frames)} 帧测试图像")
    
    # ========== 步骤 5: 处理每一帧 ==========
    print("\n4. 处理视频序列...")
    
    output_frames = []
    
    for frame_idx, frame in enumerate(frames):
        # 检测
        detections = detector.detect(frame)
        
        # 跟踪
        tracks = tracker.update(detections)
        
        # 可视化
        result_frame = visualizer.draw_tracks(
            frame.copy(),
            tracks,
            draw_history=True  # 绘制轨迹历史
        )
        
        # 添加信息文本
        info_text = f"Frame: {frame_idx} | Detections: {len(detections)} | Tracks: {len(tracks)}"
        cv2.putText(
            result_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        output_frames.append(result_frame)
        
        if (frame_idx + 1) % 10 == 0:
            print(f"  处理进度: {frame_idx + 1}/{len(frames)} 帧")
    
    print("  ✓ 视频序列处理完成")
    
    # ========== 步骤 6: 保存结果 ==========
    print("\n5. 保存结果...")
    
    # 保存为视频
    output_video_path = "rfdetr_iou_tracking_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (w, h))
    
    for frame in output_frames:
        out.write(frame)
    out.release()
    
    print(f"  ✓ 结果视频已保存到: {output_video_path}")
    
    # 保存示例帧
    cv2.imwrite("rfdetr_iou_frame_0.jpg", output_frames[0])
    cv2.imwrite("rfdetr_iou_frame_mid.jpg", output_frames[len(output_frames)//2])
    cv2.imwrite("rfdetr_iou_frame_last.jpg", output_frames[-1])
    print("  ✓ 示例帧已保存")


def example_rfdetr_with_bytetrack():
    """
    示例 2: RF-DETR 检测 + ByteTrack 跟踪器
    
    本示例展示如何使用 VisionPipeline 简化流程，使用 RF-DETR 和 ByteTrack。
    ByteTrack 是一种高性能跟踪算法，适合复杂场景。
    """
    print("=" * 70)
    print("示例 2: RF-DETR 检测 + ByteTrack 跟踪器")
    print("=" * 70)
    
    # ========== 步骤 1: 使用 VisionPipeline 简化流程 ==========
    print("\n1. 初始化视觉管道（RF-DETR + ByteTrack）...")
    
    # VisionPipeline 可以统一管理检测器和跟踪器，简化使用
    pipeline = VisionPipeline({
        "detector_config": {
            "model_type": "rfdetr",
            "conf_threshold": 0.5,
            "device": "cpu"
        },
        "tracker_config": {
            "tracker_type": "bytetrack",  # 使用 ByteTrack
            "track_thresh": 0.5,          # ByteTrack 置信度阈值
            "track_buffer": 30,           # 跟踪缓冲区
            "match_thresh": 0.8,          # 匹配阈值
            "max_age": 30,
            "min_hits": 3
        },
        "enable_tracking": True
    })
    
    if not pipeline.initialize():
        print("✗ 管道初始化失败！")
        print("  提示: 请确保已安装 rfdetr: pip install rfdetr supervision")
        return
    
    print("  ✓ 视觉管道初始化成功")
    
    # ========== 步骤 2: 初始化可视化器 ==========
    visualizer = Visualizer({
        "show_labels": True,
        "show_confidences": True,
        "show_track_ids": True,
        "line_thickness": 2
    })
    
    # ========== 步骤 3: 处理视频文件或摄像头 ==========
    print("\n2. 处理视频...")
    print("  提示: 可以修改 video_path 为视频文件路径，或使用 0 使用摄像头")
    
    video_path = 0  # 0 表示使用摄像头，或改为视频文件路径
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"✗ 无法打开视频源: {video_path}")
        print("  提示: 如果使用摄像头，请确保摄像头已连接")
        return
    
    print("  ✓ 视频源打开成功")
    print("\n  开始处理... (按 'q' 键退出)")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # ========== 步骤 4: 处理当前帧 ==========
        # pipeline.process() 会自动进行检测和跟踪
        results = pipeline.process(frame)
        tracks = results["tracks"]
        
        # ========== 步骤 5: 可视化结果 ==========
        result_frame = visualizer.draw_tracks(frame, tracks, draw_history=True)
        
        # 添加信息文本
        info_text = f"Frame: {frame_count} | Tracks: {len(tracks)}"
        cv2.putText(
            result_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        # 显示结果
        cv2.imshow("RF-DETR + ByteTrack Tracking", result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"  已处理 {frame_count} 帧，当前跟踪 {len(tracks)} 个目标")
    
    # ========== 步骤 6: 清理资源 ==========
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n  ✓ 处理完成，共处理 {frame_count} 帧")


def example_rfdetr_tracking_with_video_file():
    """
    示例 3: 使用视频文件进行 RF-DETR 跟踪
    
    本示例展示如何处理视频文件，进行检测和跟踪，并保存结果。
    """
    print("=" * 70)
    print("示例 3: RF-DETR 视频文件跟踪")
    print("=" * 70)
    
    import sys
    
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("\n使用方法:")
        print("  python rfdetr_tracking.py <video_path> [output_path]")
        print("\n示例:")
        print("  python rfdetr_tracking.py input.mp4")
        print("  python rfdetr_tracking.py input.mp4 output.mp4")
        return
    
    input_video = sys.argv[1]
    output_video = sys.argv[2] if len(sys.argv) > 2 else "rfdetr_tracking_output.mp4"
    
    # ========== 步骤 1: 初始化管道 ==========
    print(f"\n1. 初始化视觉管道...")
    
    pipeline = VisionPipeline({
        "detector_config": {
            "model_type": "rfdetr",
            "conf_threshold": 0.5,
            "device": "cpu"
        },
        "tracker_config": {
            "tracker_type": "bytetrack",
            "max_age": 30,
            "min_hits": 3
        },
        "enable_tracking": True
    })
    
    if not pipeline.initialize():
        print("✗ 管道初始化失败！")
        return
    
    print("  ✓ 管道初始化成功")
    
    # ========== 步骤 2: 打开视频文件 ==========
    print(f"\n2. 打开视频文件: {input_video}")
    
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        print(f"✗ 无法打开视频: {input_video}")
        return
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  ✓ 视频信息:")
    print(f"    分辨率: {width}x{height}")
    print(f"    帧率: {fps:.2f} FPS")
    print(f"    总帧数: {total_frames}")
    
    # ========== 步骤 3: 创建输出视频写入器 ==========
    print(f"\n3. 创建输出视频: {output_video}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("✗ 无法创建输出视频文件")
        cap.release()
        return
    
    print("  ✓ 输出视频写入器准备就绪")
    
    # ========== 步骤 4: 初始化可视化器 ==========
    visualizer = Visualizer({
        "show_labels": True,
        "show_confidences": True,
        "show_track_ids": True,
        "line_thickness": 2
    })
    
    # ========== 步骤 5: 处理视频帧 ==========
    print("\n4. 处理视频帧...")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理当前帧
        results = pipeline.process(frame)
        tracks = results["tracks"]
        
        # 可视化
        result_frame = visualizer.draw_tracks(frame, tracks, draw_history=True)
        
        # 添加信息文本
        info_text = f"Frame: {frame_count}/{total_frames} | Tracks: {len(tracks)}"
        cv2.putText(
            result_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # 写入输出视频
        out.write(result_frame)
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"  进度: {frame_count}/{total_frames} ({progress:.1f}%) | "
                  f"跟踪: {len(tracks)} 个目标")
    
    # ========== 步骤 6: 清理资源 ==========
    cap.release()
    out.release()
    
    print(f"\n  ✓ 处理完成！")
    print(f"    输入: {input_video}")
    print(f"    输出: {output_video}")
    print(f"    处理帧数: {frame_count}")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("RF-DETR 跟踪示例")
    print("=" * 70)
    print("\n本示例展示如何使用 RF-DETR 检测器进行目标跟踪")
    print("=" * 70)
    
    import sys
    
    try:
        # 示例 1: RF-DETR + IoU 跟踪器
        example_rfdetr_with_iou_tracker()
        
        print("\n" + "-" * 70)
        
        # 示例 2: RF-DETR + ByteTrack（如果提供了视频路径则处理视频文件）
        if len(sys.argv) > 1:
            example_rfdetr_tracking_with_video_file()
        else:
            print("\n提示: 可以传入视频路径作为参数来处理视频文件")
            print("例如: python rfdetr_tracking.py your_video.mp4")
            print("\n或者运行示例 2 使用摄像头（需要取消注释）:")
            # example_rfdetr_with_bytetrack()  # 取消注释以使用摄像头
            
    except KeyboardInterrupt:
        print("\n\n用户中断")
    except Exception as e:
        print(f"\n\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()

