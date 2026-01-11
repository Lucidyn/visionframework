"""
Vision Framework 高级功能使用示例

本文件包含 Vision Framework 的高级功能示例，包括：
- ROI（感兴趣区域）检测和过滤
- 对象计数
- 性能监控
- 结果导出（JSON、CSV、COCO 格式）
- 视频处理工具

这些示例适合需要更复杂功能的场景。
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from visionframework import (
    VisionPipeline, Visualizer, ResultExporter,
    PerformanceMonitor, VideoProcessor, VideoWriter,
    ROIDetector, Counter, Config
)


def example_roi_detection():
    """
    示例 1: ROI 检测和过滤
    
    本示例展示如何：
    1. 定义多个 ROI（感兴趣区域）
    2. 检测和跟踪对象
    3. 根据 ROI 过滤跟踪结果
    4. 可视化 ROI 和过滤后的结果
    
    适用于需要关注特定区域的场景，如区域监控、人流分析等。
    """
    print("=" * 70)
    print("示例 1: ROI 检测和过滤")
    print("=" * 70)
    
    # ========== 步骤 1: 创建 ROI 检测器配置 ==========
    # ROI 配置可以包含多个区域，每个区域有名称、类型和坐标点
    roi_config = {
        "rois": [
            {
                "name": "zone1",              # ROI 名称，用于后续引用
                "type": "rectangle",          # 区域类型：rectangle（矩形）
                "points": [(100, 100), (400, 300)]  # 矩形的两个对角点坐标
            },
            {
                "name": "zone2",
                "type": "polygon",            # 区域类型：polygon（多边形）
                "points": [(500, 200), (700, 200), (700, 400), (500, 400)]  # 多边形的顶点坐标
            }
        ],
        "check_center": True  # 检查边界框中心点是否在 ROI 内（而不是整个边界框）
    }
    
    # ========== 步骤 2: 初始化 ROI 检测器 ==========
    roi_detector = ROIDetector(roi_config)
    if not roi_detector.initialize():
        print("✗ ROI 检测器初始化失败")
        return
    
    print("  ✓ ROI 检测器初始化成功")
    print(f"  ✓ 定义了 {len(roi_config['rois'])} 个 ROI 区域")
    
    # ========== 步骤 3: 初始化检测和跟踪管道 ==========
    pipeline = VisionPipeline({
        "detector_config": {"model_path": "yolov8n.pt"},  # 使用 YOLO 检测器
        "enable_tracking": True  # 启用跟踪功能
    })
    
    if not pipeline.initialize():
        print("✗ 管道初始化失败")
        return
    
    print("  ✓ 检测和跟踪管道初始化成功")
    
    # ========== 步骤 4: 处理图像 ==========
    # 注意: 请将 "path/to/your/image.jpg" 替换为实际的图像路径
    image_path = "path/to/your/image.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"✗ 无法加载图像: {image_path}")
        print("  提示: 请提供有效的图像路径")
        return
    
    print(f"  ✓ 图像加载成功: {image_path}")
    
    # ========== 步骤 5: 运行检测和跟踪 ==========
    # process() 方法返回包含检测和跟踪结果的字典
    results = pipeline.process(image)
    tracks = results["tracks"]  # 获取所有跟踪结果
    
    print(f"  ✓ 检测和跟踪完成，共 {len(tracks)} 个跟踪目标")
    
    # ========== 步骤 6: 根据 ROI 过滤跟踪结果 ==========
    # filter_tracks_by_roi() 方法返回指定 ROI 内的跟踪对象
    zone1_tracks = roi_detector.filter_tracks_by_roi(tracks, "zone1")
    zone2_tracks = roi_detector.filter_tracks_by_roi(tracks, "zone2")
    
    print(f"\n  过滤结果:")
    print(f"    总跟踪数: {len(tracks)}")
    print(f"    Zone1 内: {len(zone1_tracks)} 个对象")
    print(f"    Zone2 内: {len(zone2_tracks)} 个对象")
    
    # ========== 步骤 7: 可视化结果 ==========
    visualizer = Visualizer()
    
    # 绘制所有跟踪结果
    result_image = visualizer.draw_tracks(image, tracks)
    
    # 绘制 ROI 区域
    # 遍历所有 ROI 并在图像上绘制
    for roi in roi_detector.get_rois():
        if roi.type == "rectangle":
            # 绘制矩形 ROI
            x1, y1 = map(int, roi.points[0])
            x2, y2 = map(int, roi.points[1])
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色矩形框
            cv2.putText(result_image, roi.name, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # ROI 名称
        elif roi.type == "polygon":
            # 绘制多边形 ROI（可以自行添加）
            pass
    
    # 保存结果
    cv2.imwrite("output_roi.jpg", result_image)
    print(f"\n  ✓ 结果已保存到: output_roi.jpg")


def example_counting():
    """
    示例 2: 对象计数
    
    本示例展示如何：
    1. 定义计数区域（ROI）
    2. 统计进入、离开和停留在 ROI 内的对象数量
    3. 实时显示计数信息
    
    适用于人流统计、车辆计数、区域监控等场景。
    """
    print("=" * 70)
    print("示例 2: 对象计数")
    print("=" * 70)
    
    # ========== 步骤 1: 创建计数区域配置 ==========
    roi_config = {
        "rois": [
            {
                "name": "entrance",  # 入口区域
                "type": "rectangle",
                "points": [(200, 100), (600, 400)]  # 定义入口区域范围
            }
        ]
    }
    
    # ========== 步骤 2: 创建计数器 ==========
    # Counter 需要 ROI 检测器来工作
    counter = Counter({
        "roi_detector": roi_config,  # ROI 配置
        "count_entering": True,      # 统计进入的对象
        "count_exiting": True,       # 统计离开的对象
        "count_inside": True         # 统计当前在 ROI 内的对象
    })
    
    if not counter.initialize():
        print("✗ 计数器初始化失败")
        return
    
    print("  ✓ 计数器初始化成功")
    
    # ========== 步骤 3: 初始化检测和跟踪管道 ==========
    pipeline = VisionPipeline({
        "detector_config": {"model_path": "yolov8n.pt"},
        "enable_tracking": True
    })
    
    if not pipeline.initialize():
        print("✗ 管道初始化失败")
        return
    
    print("  ✓ 管道初始化成功")
    
    # ========== 步骤 4: 初始化可视化器 ==========
    visualizer = Visualizer()
    
    # ========== 步骤 5: 处理视频 ==========
    # 注意: 请将 "path/to/your/video.mp4" 替换为实际的视频路径
    video_path = "path/to/your/video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"✗ 无法打开视频: {video_path}")
        print("  提示: 请提供有效的视频路径，或使用 0 使用摄像头")
        return
    
    print(f"  ✓ 视频打开成功: {video_path}")
    print("\n  开始处理视频... (按 'q' 键退出)")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # ========== 步骤 6: 处理当前帧 ==========
        # 运行检测和跟踪
        results = pipeline.process(frame)
        tracks = results["tracks"]
        
        # ========== 步骤 7: 统计计数 ==========
        # count_tracks() 方法返回每个 ROI 的计数信息
        counts = counter.count_tracks(tracks)
        
        # ========== 步骤 8: 可视化结果 ==========
        # 绘制跟踪结果
        result_frame = visualizer.draw_tracks(frame, tracks)
        
        # 绘制 ROI 和计数信息
        for roi_name, count_info in counts.items():
            # 获取 ROI 对象
            roi = counter.roi_detector.get_roi_by_name(roi_name)
            
            if roi and roi.type == "rectangle":
                # 绘制 ROI 矩形
                x1, y1 = map(int, roi.points[0])
                x2, y2 = map(int, roi.points[1])
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 显示计数信息
                # count_info 包含：
                # - total_entered: 累计进入数量
                # - total_exited: 累计离开数量
                # - current_inside: 当前在 ROI 内的数量
                text = f"{roi_name}: Enter={count_info['total_entered']}, " \
                       f"Exit={count_info['total_exited']}, " \
                       f"Inside={count_info['current_inside']}"
                cv2.putText(result_frame, text, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow("Counting", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        
        # 每30帧打印一次计数信息
        if frame_count % 30 == 0:
            print(f"  帧 {frame_count}: {counts}")
    
    # ========== 步骤 9: 清理和打印最终统计 ==========
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n  最终计数统计:")
    final_counts = counter.get_counts()
    for roi_name, counts in final_counts.items():
        print(f"    {roi_name}: {counts}")


def example_performance_monitoring():
    """
    示例 3: 性能监控
    
    本示例展示如何：
    1. 监控检测和可视化的处理时间
    2. 计算实时 FPS（帧率）
    3. 显示性能统计信息
    
    适用于需要优化性能或监控系统负载的场景。
    """
    print("=" * 70)
    print("示例 3: 性能监控")
    print("=" * 70)
    
    # ========== 步骤 1: 初始化管道 ==========
    pipeline = VisionPipeline({
        "detector_config": {"model_path": "yolov8n.pt"},
        "enable_tracking": True
    })
    
    if not pipeline.initialize():
        print("✗ 管道初始化失败")
        return
    
    print("  ✓ 管道初始化成功")
    
    # ========== 步骤 2: 初始化性能监控器 ==========
    # PerformanceMonitor 用于监控处理性能
    # window_size: 滑动窗口大小，用于计算平均 FPS
    monitor = PerformanceMonitor(window_size=30)
    monitor.start()  # 开始计时
    
    print("  ✓ 性能监控器初始化成功")
    
    # ========== 步骤 3: 初始化可视化器 ==========
    visualizer = Visualizer()
    
    # ========== 步骤 4: 处理视频 ==========
    video_path = "path/to/your/video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"✗ 无法打开视频: {video_path}")
        return
    
    print(f"  ✓ 视频打开成功: {video_path}")
    print("\n  开始处理并监控性能... (按 'q' 键退出)")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # ========== 步骤 5: 测量检测时间 ==========
        import time
        start = time.time()
        results = pipeline.process(frame)
        detection_time = time.time() - start
        
        tracks = results["tracks"]
        
        # ========== 步骤 6: 测量可视化时间 ==========
        start = time.time()
        result_frame = visualizer.draw_tracks(frame, tracks)
        viz_time = time.time() - start
        
        # ========== 步骤 7: 记录性能数据 ==========
        # 记录检测时间和可视化时间
        monitor.record_detection_time(detection_time)
        monitor.record_visualization_time(viz_time)
        monitor.tick()  # 更新帧计数
        
        # ========== 步骤 8: 显示性能信息 ==========
        # 获取当前 FPS
        fps = monitor.get_current_fps()
        
        # 在图像上显示 FPS
        cv2.putText(result_frame, f"FPS: {fps:.2f}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示处理时间
        cv2.putText(result_frame, f"Detection: {detection_time*1000:.1f}ms",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Visualization: {viz_time*1000:.1f}ms",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Performance", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    # ========== 步骤 9: 打印性能摘要 ==========
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n  性能统计摘要:")
    monitor.print_summary()  # 打印详细的性能统计信息


def example_result_export():
    """
    示例 4: 结果导出
    
    本示例展示如何将检测和跟踪结果导出为不同格式：
    - JSON 格式：便于程序读取和处理
    - CSV 格式：便于 Excel 等工具分析
    - COCO 格式：标准的目标检测数据集格式
    
    适用于需要保存结果用于后续分析或数据集构建的场景。
    """
    print("=" * 70)
    print("示例 4: 结果导出")
    print("=" * 70)
    
    # ========== 步骤 1: 初始化管道 ==========
    pipeline = VisionPipeline({
        "detector_config": {"model_path": "yolov8n.pt"},
        "enable_tracking": True
    })
    
    if not pipeline.initialize():
        print("✗ 管道初始化失败")
        return
    
    print("  ✓ 管道初始化成功")
    
    # ========== 步骤 2: 初始化结果导出器 ==========
    exporter = ResultExporter()
    print("  ✓ 结果导出器初始化成功")
    
    # ========== 步骤 3: 处理图像 ==========
    image_path = "path/to/your/image.jpg"
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"✗ 无法加载图像: {image_path}")
        return
    
    print(f"  ✓ 图像加载成功: {image_path}")
    
    # ========== 步骤 4: 运行检测和跟踪 ==========
    results = pipeline.process(image)
    detections = results["detections"]
    tracks = results["tracks"]
    
    print(f"  ✓ 检测到 {len(detections)} 个对象，跟踪到 {len(tracks)} 个目标")
    
    # ========== 步骤 5: 导出为 JSON 格式 ==========
    # JSON 格式便于程序读取和处理
    exporter.export_detections_to_json(
        detections,
        "output_detections.json",
        metadata={"image_path": image_path}  # 可以添加额外的元数据
    )
    print("  ✓ 检测结果已导出到: output_detections.json")
    
    exporter.export_tracks_to_json(
        tracks,
        "output_tracks.json",
        metadata={"image_path": image_path}
    )
    print("  ✓ 跟踪结果已导出到: output_tracks.json")
    
    # ========== 步骤 6: 导出为 CSV 格式 ==========
    # CSV 格式便于 Excel 等工具分析
    exporter.export_detections_to_csv(detections, "output_detections.csv")
    print("  ✓ 检测结果已导出到: output_detections.csv")
    
    exporter.export_tracks_to_csv(tracks, "output_tracks.csv")
    print("  ✓ 跟踪结果已导出到: output_tracks.csv")
    
    # ========== 步骤 7: 导出为 COCO 格式 ==========
    # COCO 格式是标准的目标检测数据集格式
    h, w = image.shape[:2]  # 获取图像尺寸
    
    exporter.export_to_coco_format(
        detections,
        image_id=0,  # 图像 ID
        image_info={
            "width": w,
            "height": h,
            "file_name": image_path
        },
        output_path="output_coco.json"
    )
    print("  ✓ COCO 格式结果已导出到: output_coco.json")
    
    print("\n  所有结果已成功导出！")


def example_video_processing():
    """
    示例 5: 视频处理工具
    
    本示例展示如何使用视频处理工具：
    - 使用 VideoProcessor 读取视频
    - 使用 VideoWriter 写入视频
    - 使用回调函数处理每一帧
    
    适用于需要批量处理视频或自定义处理流程的场景。
    """
    print("=" * 70)
    print("示例 5: 视频处理工具")
    print("=" * 70)
    
    # ========== 步骤 1: 初始化管道 ==========
    pipeline = VisionPipeline({
        "detector_config": {"model_path": "yolov8n.pt"},
        "enable_tracking": True
    })
    
    if not pipeline.initialize():
        print("✗ 管道初始化失败")
        return
    
    print("  ✓ 管道初始化成功")
    
    # ========== 步骤 2: 初始化可视化器 ==========
    visualizer = Visualizer()
    
    # ========== 步骤 3: 定义帧处理回调函数 ==========
    # 这个函数会被 VideoProcessor 调用，处理每一帧
    def process_frame(frame, frame_num):
        """
        帧处理回调函数
        
        Args:
            frame: 当前帧图像
            frame_num: 当前帧编号
        
        Returns:
            处理后的帧图像
        """
        # 运行检测和跟踪
        results = pipeline.process(frame)
        tracks = results["tracks"]
        
        # 可视化结果
        result_frame = visualizer.draw_tracks(frame, tracks)
        
        # 可以在这里添加自定义处理逻辑
        # 例如：添加水印、调整亮度等
        
        return result_frame
    
    # ========== 步骤 4: 处理视频 ==========
    input_video = "path/to/input_video.mp4"
    output_video = "output_processed.mp4"
    
    print(f"  输入视频: {input_video}")
    print(f"  输出视频: {output_video}")
    
    # 使用 VideoProcessor 处理视频
    # 注意: 这里需要导入 process_video 函数
    # 实际使用时，可以使用 VideoProcessor 和 VideoWriter 类
    from visionframework.utils.video_utils import process_video
    
    success = process_video(
        input_path=input_video,
        output_path=output_video,
        frame_callback=process_frame,  # 帧处理回调函数
        start_frame=0,                 # 起始帧（可选）
        end_frame=None,                 # 结束帧（None 表示处理到结尾）
        skip_frames=0                   # 跳过的帧数（0 表示不跳过）
    )
    
    if success:
        print(f"\n  ✓ 视频处理完成，已保存到: {output_video}")
    else:
        print("\n  ✗ 视频处理失败")


if __name__ == "__main__":
    """
    主函数
    
    运行此文件时，可以选择运行哪个示例。
    默认情况下所有示例都被注释，需要取消注释才能运行。
    """
    print("\n" + "=" * 70)
    print("Vision Framework - 高级功能示例")
    print("=" * 70)
    print("\n提示: 请先修改示例中的图像/视频路径，然后取消注释要运行的示例")
    print("\n可用示例:")
    print("1. example_roi_detection() - ROI 检测和过滤")
    print("2. example_counting() - 对象计数")
    print("3. example_performance_monitoring() - 性能监控")
    print("4. example_result_export() - 结果导出")
    print("5. example_video_processing() - 视频处理工具")
    print("\n" + "=" * 70)
    
    # 取消注释下面的行来运行对应的示例
    # example_roi_detection()
    # example_counting()
    # example_performance_monitoring()
    # example_result_export()
    # example_video_processing()
