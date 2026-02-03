"""
12_result_export.py

结果导出示例：
- 使用 ResultExporter 导出处理结果到不同格式
- 支持 JSON、CSV、YAML 等格式
- 导出配置和性能指标

注意：
- 需要提前准备模型权重（如 yolov8n.pt），并放在当前工作目录或指定路径。
"""

import os
import cv2
import numpy as np

from visionframework import (
    YOLODetector, ByteTracker, PoseEstimator,
    ResultExporter, Visualizer, Detection, Track, Pose, KeyPoint
)


def create_dummy_results():
    """
    创建虚拟的处理结果，用于测试导出功能。
    """
    # 创建虚拟检测结果
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
    
    # 创建虚拟跟踪结果
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
    
    # 创建虚拟姿态估计结果
    poses = [
        Pose(
            bbox=(100, 100, 200, 300),
            keypoints=[
                KeyPoint(keypoint_id=0, keypoint_name="nose", x=150, y=120, confidence=0.9),
                KeyPoint(keypoint_id=1, keypoint_name="left_eye", x=130, y=110, confidence=0.8),
                KeyPoint(keypoint_id=2, keypoint_name="right_eye", x=170, y=110, confidence=0.8),
                KeyPoint(keypoint_id=3, keypoint_name="neck", x=150, y=180, confidence=0.85),
                KeyPoint(keypoint_id=4, keypoint_name="left_shoulder", x=120, y=220, confidence=0.8),
                KeyPoint(keypoint_id=5, keypoint_name="right_shoulder", x=180, y=220, confidence=0.8)
            ],
            confidence=0.85,
            pose_id=1
        )
    ]
    
    # 创建结果字典
    results = {
        "detections": detections,
        "tracks": tracks,
        "poses": poses,
        "metadata": {
            "image_width": 640,
            "image_height": 480,
            "processing_time": 0.123,
            "timestamp": "2024-01-01T12:00:00"
        }
    }
    
    return results


def export_to_json(exporter, results, output_dir):
    """
    导出结果到 JSON 格式。
    """
    output_path = os.path.join(output_dir, "detections.json")
    exporter.export_detections_to_json(results["detections"], output_path)
    print(f"检测结果已导出到 JSON 文件: {output_path}")
    
    output_path = os.path.join(output_dir, "tracks.json")
    exporter.export_tracks_to_json(results["tracks"], output_path)
    print(f"跟踪结果已导出到 JSON 文件: {output_path}")


def export_to_csv(exporter, results, output_dir):
    """
    导出结果到 CSV 格式。
    """
    output_path = os.path.join(output_dir, "detections.csv")
    exporter.export_detections_to_csv(results["detections"], output_path)
    print(f"检测结果已导出到 CSV 文件: {output_path}")
    
    output_path = os.path.join(output_dir, "tracks.csv")
    exporter.export_tracks_to_csv(results["tracks"], output_path)
    print(f"跟踪结果已导出到 CSV 文件: {output_path}")


def export_with_real_detections():
    """
    使用真实的检测结果进行导出。
    """
    print("使用真实的检测结果进行导出...")
    
    # 创建检测器配置
    detector_config = {
        "model_path": "yolov8n.pt",  # 可替换为你的模型路径
        "device": "auto",
        "conf_threshold": 0.25,
    }
    
    # 初始化检测器
    detector = YOLODetector(detector_config)
    if not detector.initialize():
        print("YOLODetector 初始化失败，请检查模型路径和依赖。")
        return None
    
    # 创建测试图像
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.rectangle(test_image, (300, 150, 400, 250), (255, 255, 255), -1)
    
    # 执行检测
    detections = detector.detect(test_image)
    print(f"检测到 {len(detections)} 个目标")
    
    # 创建结果字典
    results = {
        "detections": detections,
        "tracks": [],
        "poses": [],
        "metadata": {
            "image_width": 640,
            "image_height": 480,
            "processing_time": 0.123,
            "timestamp": "2024-01-01T12:00:00"
        }
    }
    
    return results


def main():
    """
    主函数。
    """
    # 创建导出目录
    output_dir = "exports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建导出目录: {output_dir}")
    
    # 初始化 ResultExporter
    exporter = ResultExporter()
    print("ResultExporter 初始化成功!")
    
    # 测试 1: 使用虚拟结果导出
    print("\n测试 1: 使用虚拟结果导出")
    dummy_results = create_dummy_results()
    
    # 导出到不同格式
    export_to_json(exporter, dummy_results, output_dir)
    export_to_csv(exporter, dummy_results, output_dir)
    
    # 测试 2: 使用真实检测结果导出
    print("\n测试 2: 使用真实检测结果导出")
    real_results = export_with_real_detections()
    
    if real_results:
        # 导出真实结果
        real_output_dir = os.path.join(output_dir, "real")
        if not os.path.exists(real_output_dir):
            os.makedirs(real_output_dir)
        
        export_to_json(exporter, real_results, real_output_dir)
        export_to_csv(exporter, real_results, real_output_dir)
    
    print("\n所有导出测试完成!")


if __name__ == "__main__":
    main()
