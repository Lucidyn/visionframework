"""
视频跟踪示例

本示例展示如何使用 Vision Framework 处理视频进行目标跟踪。
支持摄像头实时处理和视频文件处理两种模式。
"""

import cv2
import sys
from pathlib import Path

# 将父目录添加到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from visionframework import VisionPipeline, Visualizer, Config


def main():
    """
    主函数：视频跟踪处理
    
    本函数展示了完整的视频跟踪流程：
    1. 配置管道参数
    2. 初始化检测和跟踪模块
    3. 打开视频源（文件或摄像头）
    4. 逐帧处理并可视化
    5. 显示结果
    """
    # ========== 步骤 1: 配置管道参数 ==========
    # 使用 Config 工具获取默认配置，然后根据需要修改
    config = Config.get_default_pipeline_config()
    
    # 检测器配置
    config["detector_config"]["model_path"] = "yolov8n.pt"  # YOLO 模型路径
    config["detector_config"]["conf_threshold"] = 0.25       # 置信度阈值 25%
    
    # 跟踪器配置
    # max_age: 目标丢失后保留的最大帧数，超过此值会删除跟踪
    config["tracker_config"]["max_age"] = 30
    
    # min_hits: 确认跟踪所需的最小命中次数，避免误跟踪
    config["tracker_config"]["min_hits"] = 3
    
    # ========== 步骤 2: 初始化管道 ==========
    print("正在初始化管道...")
    pipeline = VisionPipeline(config)
    
    # 检查初始化是否成功
    if not pipeline.initialize():
        print("管道初始化失败！")
        print("可能的原因：")
        print("  1. 模型文件不存在或下载失败")
        print("  2. 缺少必要的依赖（如 ultralytics）")
        print("  3. 设备配置错误（如指定了不存在的 GPU）")
        return
    
    print("✓ 管道初始化成功")
    
    # ========== 步骤 3: 配置可视化器 ==========
    # 可视化器用于在视频帧上绘制检测和跟踪结果
    visualizer = Visualizer({
        "show_labels": True,        # 显示类别标签（如 "person", "car"）
        "show_confidences": True,   # 显示检测置信度
        "show_track_ids": True,     # 显示跟踪ID（每个目标有唯一ID）
        "line_thickness": 2         # 绘制线条的粗细
    })
    
    # ========== 步骤 4: 打开视频源 ==========
    # 方式1: 使用摄像头（实时处理）
    # video_path = 0  # 0 表示使用默认摄像头
    
    # 方式2: 使用视频文件
    video_path = "path/to/your/video.mp4"  # 请替换为实际的视频路径
    
    # 创建视频捕获对象
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频源是否成功打开
    if not cap.isOpened():
        print(f"无法打开视频源: {video_path}")
        print("\n提示:")
        print("  - 如果使用摄像头，请确保摄像头已连接")
        print("  - 如果使用视频文件，请检查文件路径是否正确")
        print("  - 可以尝试将 video_path 改为 0 使用摄像头")
        return
    
    # 获取视频信息（仅对视频文件有效）
    if isinstance(video_path, str):
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"✓ 视频信息:")
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps:.2f} FPS")
        print(f"  总帧数: {total_frames}")
    
    # ========== 步骤 5: 处理视频帧 ==========
    print("\n开始处理视频... (按 'q' 键退出)")
    
    frame_count = 0  # 帧计数器
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        
        # 检查是否成功读取
        if not ret:
            print("\n视频读取完成或出错")
            break
        
        # 使用管道处理当前帧
        # process() 方法会：
        # 1. 使用检测器检测目标
        # 2. 使用跟踪器更新跟踪状态
        # 3. 返回包含检测和跟踪结果的字典
        results = pipeline.process(frame)
        tracks = results["tracks"]  # 获取跟踪结果列表
        
        # 可视化跟踪结果
        # draw_tracks 会：
        # 1. 绘制每个目标的边界框
        # 2. 显示跟踪ID和类别标签
        # 3. 绘制轨迹历史（如果 draw_history=True）
        result_frame = visualizer.draw_tracks(frame, tracks, draw_history=True)
        
        # ========== 步骤 6: 添加信息文本 ==========
        # 在图像上显示当前帧的统计信息
        info_text = f"Frame: {frame_count} | Tracks: {len(tracks)}"
        cv2.putText(
            result_frame,           # 目标图像
            info_text,              # 要显示的文本
            (10, 30),               # 文本位置 (x, y)
            cv2.FONT_HERSHEY_SIMPLEX,  # 字体类型
            1,                      # 字体大小
            (0, 255, 0),            # 颜色 (B, G, R)
            2                       # 线条粗细
        )
        
        # ========== 步骤 7: 显示结果 ==========
        # 在窗口中显示处理后的帧
        cv2.imshow("Video Tracking", result_frame)
        
        # 检查用户输入
        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n用户中断处理")
            break
        
        # 更新帧计数
        frame_count += 1
        
        # 每30帧打印一次进度（可选）
        if frame_count % 30 == 0:
            print(f"  已处理 {frame_count} 帧，当前跟踪 {len(tracks)} 个目标")
    
    # ========== 步骤 8: 清理资源 ==========
    # 释放视频捕获对象
    cap.release()
    
    # 关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()
    
    # 打印处理统计
    print(f"\n✓ 处理完成！")
    print(f"  总处理帧数: {frame_count}")
    
    # 提示：可以在这里添加结果保存功能
    # 例如：保存处理后的视频、导出跟踪数据等


if __name__ == "__main__":
    """
    程序入口
    
    运行此脚本会执行视频跟踪处理。
    使用前请确保：
    1. 已安装所有依赖: pip install -r requirements.txt
    2. 已准备好视频文件或连接摄像头
    3. 修改 video_path 为实际的视频路径或使用 0 使用摄像头
    """
    main()
