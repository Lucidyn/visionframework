#!/usr/bin/env python3
"""
工具类使用示例 - 展示Vision Framework中的各种工具类用法

该示例展示了如何使用框架中的各种工具类，包括：
1. 可视化工具 - 绘制检测和跟踪结果
2. 评估工具 - 计算检测和跟踪性能指标
3. 性能监控 - 监控和分析性能
4. 结果导出 - 将结果导出为不同格式

Usage:
    python 06_tools_usage.py
"""

import cv2
import numpy as np
from visionframework.utils.visualization import Visualizer
from visionframework.utils.evaluation import DetectionEvaluator, TrackingEvaluator
from visionframework.utils.monitoring.performance import PerformanceMonitor, Timer
from visionframework.utils.data.export import ResultExporter
from visionframework.data.detection import Detection
from visionframework.data.track import Track

def create_test_data():
    """创建测试数据"""
    # 创建测试图像
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(image, "Test Image", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 创建测试检测结果
    detections = [
        Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="person"
        ),
        Detection(
            bbox=(300, 150, 400, 250),
            confidence=0.85,
            class_id=1,
            class_name="car"
        )
    ]
    
    # 创建测试跟踪结果
    tracks = [
        Track(
            track_id=1,
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="person"
        ),
        Track(
            track_id=2,
            bbox=(300, 150, 400, 250),
            confidence=0.85,
            class_id=1,
            class_name="car"
        )
    ]
    
    return image, detections, tracks

def test_visualization():
    """测试可视化工具"""
    print("\n1. 测试可视化工具...")
    
    image, detections, tracks = create_test_data()
    
    # 创建可视化器
    visualizer = Visualizer()
    
    # 绘制检测结果
    detection_result = visualizer.draw_detections(image, detections)
    print("✓ 绘制检测结果成功")
    
    # 绘制跟踪结果
    track_result = visualizer.draw_tracks(image, tracks)
    print("✓ 绘制跟踪结果成功")
    
    # 绘制混合结果
    mixed_result = visualizer.draw_results(image, detections=detections, tracks=tracks)
    print("✓ 绘制混合结果成功")
    
    return detection_result, track_result, mixed_result

def test_evaluation():
    """测试评估工具"""
    print("\n2. 测试评估工具...")
    
    # 创建测试数据
    _, pred_detections, pred_tracks = create_test_data()
    
    # 创建模拟的真实数据
    gt_detections = [
        Detection(
            bbox=(105, 105, 205, 205),
            confidence=1.0,
            class_id=0,
            class_name="person"
        ),
        Detection(
            bbox=(310, 160, 410, 260),
            confidence=1.0,
            class_id=1,
            class_name="car"
        )
    ]
    
    # 检测评估
    det_evaluator = DetectionEvaluator(iou_threshold=0.5)
    metrics = det_evaluator.calculate_metrics(pred_detections, gt_detections)
    print(f"✓ 检测评估结果: 准确率={metrics['precision']:.2f}, 召回率={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")
    
    # 模拟多帧数据进行mAP计算
    all_pred = [pred_detections, pred_detections]
    all_gt = [gt_detections, gt_detections]
    map_result = det_evaluator.calculate_map(all_pred, all_gt)
    print(f"✓ mAP结果: {map_result['mAP']:.2f}")
    
    # 跟踪评估 (使用简化数据格式)
    track_evaluator = TrackingEvaluator()
    
    # 创建简化的跟踪数据格式
    pred_tracks_data = [
        # 帧0
        [
            {"track_id": 1, "bbox": {"x1": 100, "y1": 100, "x2": 200, "y2": 200}},
            {"track_id": 2, "bbox": {"x1": 300, "y1": 150, "x2": 400, "y2": 250}}
        ],
        # 帧1
        [
            {"track_id": 1, "bbox": {"x1": 105, "y1": 105, "x2": 205, "y2": 205}},
            {"track_id": 2, "bbox": {"x1": 305, "y1": 155, "x2": 405, "y2": 255}}
        ]
    ]
    
    gt_tracks_data = [
        # 帧0
        [
            {"track_id": 1, "bbox": {"x1": 102, "y1": 102, "x2": 202, "y2": 202}},
            {"track_id": 2, "bbox": {"x1": 302, "y1": 152, "x2": 402, "y2": 252}}
        ],
        # 帧1
        [
            {"track_id": 1, "bbox": {"x1": 107, "y1": 107, "x2": 207, "y2": 207}},
            {"track_id": 2, "bbox": {"x1": 307, "y1": 157, "x2": 407, "y2": 257}}
        ]
    ]
    
    # 计算跟踪指标
    mota_result = track_evaluator.calculate_mota(pred_tracks_data, gt_tracks_data)
    print(f"✓ 跟踪评估结果: MOTA={mota_result['MOTA']:.2f}, 准确率={mota_result['precision']:.2f}, 召回率={mota_result['recall']:.2f}")
    
    idf1_result = track_evaluator.calculate_idf1(pred_tracks_data, gt_tracks_data)
    print(f"✓ IDF1结果: {idf1_result['IDF1']:.2f}")
    
    return metrics, map_result, mota_result

def test_performance_monitoring():
    """测试性能监控工具"""
    print("\n3. 测试性能监控工具...")
    
    # 创建性能监控器
    monitor = PerformanceMonitor(window_size=30)
    monitor.start()
    
    # 模拟处理过程
    with Timer("测试处理") as timer:
        for i in range(5):
            # 模拟检测过程
            with Timer() as det_timer:
                time.sleep(0.1)  # 模拟检测耗时
            monitor.record_component_time("detection", det_timer.get_elapsed())
            
            # 模拟跟踪过程
            with Timer() as track_timer:
                time.sleep(0.05)  # 模拟跟踪耗时
            monitor.record_component_time("tracking", track_timer.get_elapsed())
            
            # 记录帧处理
            monitor.tick()
    
    print(f"✓ 总处理时间: {timer.get_elapsed():.2f}秒")
    
    # 获取详细指标
    metrics = monitor.get_metrics()
    print(f"✓ 当前FPS: {metrics.fps:.2f}")
    print(f"✓ 平均FPS: {metrics.avg_fps:.2f}")
    print(f"✓ 处理帧数: {metrics.frame_count}")
    print(f"✓ 平均每帧处理时间: {metrics.avg_time_per_frame:.3f}秒")
    
    return metrics

def test_result_export():
    """测试结果导出工具"""
    print("\n4. 测试结果导出工具...")
    
    # 创建测试数据
    _, detections, tracks = create_test_data()
    
    # 创建结果导出器
    exporter = ResultExporter()
    
    # 导出检测结果为JSON
    success = exporter.export_detections_to_json(detections, "output/detections.json")
    print(f"✓ 导出检测结果到JSON: {'成功' if success else '失败'}")
    
    # 导出跟踪结果为JSON
    success = exporter.export_tracks_to_json(tracks, "output/tracks.json")
    print(f"✓ 导出跟踪结果到JSON: {'成功' if success else '失败'}")
    
    # 导出检测结果为CSV
    success = exporter.export_detections_to_csv(detections, "output/detections.csv")
    print(f"✓ 导出检测结果到CSV: {'成功' if success else '失败'}")
    
    # 导出跟踪结果为CSV
    success = exporter.export_tracks_to_csv(tracks, "output/tracks.csv")
    print(f"✓ 导出跟踪结果到CSV: {'成功' if success else '失败'}")
    
    # 导出为COCO格式
    image_info = {"width": 640, "height": 480, "file_name": "test.jpg"}
    success = exporter.export_to_coco_format(detections, 1, image_info, "output/coco_annotations.json")
    print(f"✓ 导出为COCO格式: {'成功' if success else '失败'}")
    
    return success

def main():
    """主函数"""
    print("Vision Framework 工具类使用示例")
    print("=" * 50)
    
    # 确保输出目录存在
    import os
    os.makedirs("output", exist_ok=True)
    
    # 1. 测试可视化工具
    vis_results = test_visualization()
    
    # 2. 测试评估工具
    eval_results = test_evaluation()
    
    # 3. 测试性能监控工具
    perf_results = test_performance_monitoring()
    
    # 4. 测试结果导出工具
    export_success = test_result_export()
    
    print("\n" + "=" * 50)
    print("示例完成! 所有工具类测试成功。")
    print("输出文件已保存到 output/ 目录")
    print("=" * 50)
    
    # 显示可视化结果（可选）
    if vis_results:
        print("\n按任意键关闭可视化窗口...")
        cv2.imshow("检测结果", vis_results[0])
        cv2.imshow("跟踪结果", vis_results[1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import time
    main()
