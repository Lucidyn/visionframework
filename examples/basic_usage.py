"""
Vision Framework 基本使用示例

本文件包含 Vision Framework 的基本使用示例，适合初学者快速上手。
每个示例都包含详细的中文注释，说明每一步的作用和参数含义。
"""

import cv2
import sys
from pathlib import Path

# 将父目录添加到 Python 路径，以便导入 visionframework 模块
sys.path.insert(0, str(Path(__file__).parent.parent))

from visionframework import Detector, Tracker, VisionPipeline, Visualizer, Config


def example_detection_only():
    """
    示例 1: 仅使用检测功能
    
    这个示例展示如何使用检测器单独进行目标检测，不涉及跟踪功能。
    适用于只需要检测结果的场景，如静态图像分析、批量图像处理等。
    """
    print("=" * 50)
    print("示例 1: 仅检测功能")
    print("=" * 50)
    
    # 步骤 1: 获取默认检测器配置
    # Config.get_default_detector_config() 返回包含所有默认参数的配置字典
    detector_config = Config.get_default_detector_config()
    
    # 步骤 2: 自定义配置参数
    # model_path: 模型文件路径，首次使用会自动从网络下载
    # conf_threshold: 置信度阈值，低于此值的检测结果会被过滤
    detector_config["model_path"] = "yolov8n.pt"  # 会自动下载
    detector_config["conf_threshold"] = 0.25  # 25% 置信度阈值
    
    # 步骤 3: 创建并初始化检测器
    # Detector 是统一的检测器接口，支持 YOLO、DETR、RF-DETR 等多种模型
    detector = Detector(detector_config)
    
    # initialize() 方法会加载模型到内存，返回 True 表示成功
    if not detector.initialize():
        print("检测器初始化失败，请检查模型路径和依赖是否安装")
        return
    
    print("✓ 检测器初始化成功")
    
    # 步骤 4: 加载待检测的图像
    # 注意: 请将 "path/to/your/image.jpg" 替换为实际的图像路径
    image_path = "path/to/your/image.jpg"
    image = cv2.imread(image_path)
    
    # 检查图像是否成功加载
    if image is None:
        print(f"无法加载图像: {image_path}")
        print("请提供有效的图像路径")
        return
    
    print(f"✓ 图像加载成功，尺寸: {image.shape[1]}x{image.shape[0]}")
    
    # 步骤 5: 运行检测
    # detect() 方法返回 Detection 对象列表，每个对象包含：
    # - bbox: 边界框坐标 (x1, y1, x2, y2)
    # - confidence: 检测置信度
    # - class_id: 类别ID
    # - class_name: 类别名称
    detections = detector.detect(image)
    print(f"✓ 检测完成，发现 {len(detections)} 个对象")
    
    # 打印检测结果详情
    for i, det in enumerate(detections):
        print(f"  对象 {i+1}: {det.class_name} (置信度: {det.confidence:.2f})")
    
    # 步骤 6: 可视化检测结果
    # Visualizer 提供多种可视化方法，draw_detections 用于绘制检测框
    visualizer = Visualizer()
    result_image = visualizer.draw_detections(image, detections)
    
    # 步骤 7: 保存结果图像
    # 结果图像会包含检测框、类别标签和置信度信息
    cv2.imwrite("output_detection.jpg", result_image)
    print("✓ 结果已保存到: output_detection.jpg")


def example_tracking():
    """
    示例 2: 检测 + 跟踪
    
    这个示例展示如何使用 VisionPipeline 同时进行检测和跟踪。
    适用于视频处理、实时监控等需要跟踪目标移动轨迹的场景。
    """
    print("=" * 50)
    print("示例 2: 检测 + 跟踪")
    print("=" * 50)
    
    # 步骤 1: 获取默认管道配置
    # 管道配置包含检测器和跟踪器的所有配置参数
    pipeline_config = Config.get_default_pipeline_config()
    
    # 步骤 2: 配置检测器参数
    pipeline_config["detector_config"]["model_path"] = "yolov8n.pt"
    pipeline_config["detector_config"]["conf_threshold"] = 0.25
    
    # 步骤 3: 启用跟踪功能
    # enable_tracking=True 表示在处理时同时进行跟踪
    pipeline_config["enable_tracking"] = True
    
    # 步骤 4: 创建并初始化管道
    # VisionPipeline 是完整的检测+跟踪管道，简化了使用流程
    pipeline = VisionPipeline(pipeline_config)
    
    if not pipeline.initialize():
        print("管道初始化失败")
        return
    
    print("✓ 管道初始化成功")
    
    # 步骤 5: 初始化可视化器
    # 可视化器用于在图像上绘制检测和跟踪结果
    visualizer = Visualizer()
    
    # 步骤 6: 打开视频文件或摄像头
    # 注意: 请将 "path/to/your/video.mp4" 替换为实际的视频路径
    # 或者使用 0 来使用默认摄像头
    video_path = "path/to/your/video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        print("请提供有效的视频路径，或使用 0 使用摄像头")
        return
    
    print(f"✓ 视频打开成功: {video_path}")
    
    # 步骤 7: 逐帧处理视频
    frame_count = 0
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("视频读取完成或出错")
            break
        
        # 处理当前帧
        # process() 方法返回包含 'detections' 和 'tracks' 的字典
        results = pipeline.process(frame)
        tracks = results["tracks"]  # 获取跟踪结果
        
        # 可视化跟踪结果
        # draw_tracks 会绘制跟踪框、轨迹历史和跟踪ID
        result_frame = visualizer.draw_tracks(frame, tracks, draw_history=True)
        
        # 显示结果（可选）
        cv2.imshow("Tracking", result_frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("用户中断")
            break
        
        frame_count += 1
        
        # 每30帧打印一次进度
        if frame_count % 30 == 0:
            print(f"  已处理 {frame_count} 帧，当前跟踪 {len(tracks)} 个目标")
    
    # 步骤 8: 清理资源
    cap.release()
    cv2.destroyAllWindows()
    print(f"✓ 处理完成，共处理 {frame_count} 帧")


def example_pipeline():
    """
    示例 3: 使用完整管道
    
    这个示例展示如何自定义配置并使用完整的 VisionPipeline。
    适合需要精细控制检测和跟踪参数的场景。
    """
    print("=" * 50)
    print("示例 3: 完整管道使用")
    print("=" * 50)
    
    # 步骤 1: 创建自定义配置
    # 可以完全自定义所有参数，不依赖默认配置
    config = {
        "enable_tracking": True,  # 启用跟踪
        
        # 检测器配置
        "detector_config": {
            "model_path": "yolov8n.pt",      # 模型路径
            "conf_threshold": 0.3,          # 置信度阈值（30%）
            "device": "cpu"                  # 设备类型：cpu/cuda/mps
        },

        # 性能选项示例：可选开启批量推理与 FP16（在 cuda 下）
        # 将性能参数放在 pipeline 顶层，或直接放到 detector_config 的 "performance" 字段
        "performance": {
            "batch_inference": False,
            "use_fp16": False
        },
        
        # 跟踪器配置
        "tracker_config": {
            "max_age": 30,        # 目标丢失后保留的最大帧数
            "min_hits": 3,        # 确认跟踪所需的最小命中次数
            "iou_threshold": 0.3  # IoU匹配阈值
        }
    }
    
    # 步骤 2: 初始化管道
    pipeline = VisionPipeline(config)
    
    if not pipeline.initialize():
        print("管道初始化失败")
        return
    
    print("✓ 管道初始化成功")
    
    # 步骤 3: 配置可视化器
    # 可以自定义可视化选项
    visualizer = Visualizer({
        "show_labels": True,        # 显示类别标签
        "show_confidences": True,   # 显示置信度
        "show_track_ids": True      # 显示跟踪ID
    })
    
    # 步骤 4: 加载并处理图像
    image_path = "path/to/your/image.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"无法加载图像: {image_path}")
        return
    
    # 处理图像
    # 管道会自动进行检测和跟踪（如果启用）
    results = pipeline.process(image)
    
    # 步骤 5: 获取结果
    detections = results["detections"]  # 检测结果
    tracks = results["tracks"]          # 跟踪结果
    
    print(f"检测结果: {len(detections)} 个对象")
    print(f"跟踪结果: {len(tracks)} 个轨迹")
    
    # 步骤 6: 可视化并保存
    # draw_results 可以同时绘制检测和跟踪结果
    result_image = visualizer.draw_results(
        image,
        detections=detections,
        tracks=tracks
    )
    
    cv2.imwrite("output_pipeline.jpg", result_image)
    print("✓ 结果已保存到: output_pipeline.jpg")


def example_custom_usage():
    """
    示例 4: 自定义组件使用
    
    这个示例展示如何分别使用检测器和跟踪器，而不是使用管道。
    适合需要更灵活控制的场景，比如自定义处理流程、多阶段处理等。
    """
    print("=" * 50)
    print("示例 4: 自定义组件使用")
    print("=" * 50)
    
    # 步骤 1: 单独创建检测器
    # 这种方式可以更灵活地控制检测过程
    detector = Detector({
        "model_path": "yolov8n.pt",
        "conf_threshold": 0.25
    })
    
    if not detector.initialize():
        print("检测器初始化失败")
        return
    
    print("✓ 检测器初始化成功")
    
    # 步骤 2: 单独创建跟踪器
    # 跟踪器可以独立使用，不依赖检测器
    tracker = Tracker({
        "max_age": 30,      # 目标丢失后保留30帧
        "min_hits": 3       # 需要3次匹配才确认跟踪
    })
    
    if not tracker.initialize():
        print("跟踪器初始化失败")
        return
    
    print("✓ 跟踪器初始化成功")
    
    # 步骤 3: 加载图像
    image_path = "path/to/your/image.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"无法加载图像: {image_path}")
        return
    
    # 步骤 4: 先进行检测
    # 检测器返回 Detection 对象列表
    detections = detector.detect(image)
    print(f"✓ 检测完成，发现 {len(detections)} 个对象")
    
    # 步骤 5: 再进行跟踪
    # 跟踪器需要检测结果作为输入
    # update() 方法会根据当前检测结果更新跟踪状态
    tracks = tracker.update(detections)
    print(f"✓ 跟踪完成，当前跟踪 {len(tracks)} 个目标")
    
    # 步骤 6: 可视化结果
    visualizer = Visualizer()
    
    # 可以分别可视化检测和跟踪结果
    # 这里只可视化跟踪结果（已包含检测信息）
    result = visualizer.draw_tracks(image, tracks)
    
    cv2.imwrite("output_custom.jpg", result)
    print("✓ 结果已保存到: output_custom.jpg")
    
    # 提示: 这种方式适合需要自定义处理流程的场景
    # 例如：可以先检测，然后进行过滤，再进行跟踪


if __name__ == "__main__":
    """
    主函数
    
    运行此文件时，可以选择运行哪个示例。
    默认情况下所有示例都被注释，需要取消注释才能运行。
    """
    print("\n" + "=" * 50)
    print("Vision Framework - 基本使用示例")
    print("=" * 50)
    print("\n提示: 请先修改示例中的图像/视频路径，然后取消注释要运行的示例")
    print("\n可用示例:")
    print("1. example_detection_only() - 仅检测功能")
    print("2. example_tracking() - 检测 + 跟踪")
    print("3. example_pipeline() - 完整管道使用")
    print("4. example_custom_usage() - 自定义组件使用")
    print("\n" + "=" * 50)
    
    # 取消注释下面的行来运行对应的示例
    # example_detection_only()
    # example_tracking()
    # example_pipeline()
    # example_custom_usage()
