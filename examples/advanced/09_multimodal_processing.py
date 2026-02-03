"""
09_multimodal_processing.py

多模态处理示例：
- 结合检测、跟踪、姿态估计和特征提取
- 多模态结果融合和分析

注意：
- 需要提前准备相关模型权重，并放在当前工作目录或指定路径。
"""

import cv2
import numpy as np

from visionframework import (
    YOLODetector, ByteTracker, PoseEstimator, CLIPExtractor, ReIDExtractor,
    Visualizer, Detection, Track
)


def main() -> None:
    # 1. 创建各个组件的配置
    detector_config = {
        "model_path": "yolov8n.pt",  # 可替换为你的模型路径
        "device": "auto",
        "conf_threshold": 0.25,
    }
    
    tracker_config = {
        "track_thresh": 0.5,
        "track_buffer": 30,
        "match_thresh": 0.7,
        "frame_rate": 30,
    }
    
    pose_config = {
        "model_path": "yolov8n-pose.pt",  # 可替换为你的模型路径
        "device": "auto",
        "conf_threshold": 0.25,
    }
    
    clip_config = {
        "model_path": "ViT-B-32.pt",  # 可替换为你的模型路径
        "device": "auto",
    }
    
    reid_config = {
        "model_path": "osnet_x0_25_msmt17.pt",  # 可替换为你的模型路径
        "device": "auto",
    }

    # 2. 初始化各个组件
    print("初始化组件...")
    
    # 检测器
    detector = YOLODetector(detector_config)
    if not detector.initialize():
        print("YOLODetector 初始化失败，请检查模型路径和依赖。")
        return
    
    # 跟踪器
    tracker = ByteTracker(tracker_config)
    
    # 姿态估计器
    pose_estimator = PoseEstimator(pose_config)
    if not pose_estimator.initialize():
        print("PoseEstimator 初始化失败，请检查模型路径和依赖。")
        return
    
    # CLIP特征提取器
    clip_extractor = CLIPExtractor(clip_config)
    if not clip_extractor.initialize():
        print("CLIPExtractor 初始化失败，请检查模型路径和依赖。")
        return
    
    # ReID特征提取器
    reid_extractor = ReIDExtractor(reid_config)
    if not reid_extractor.initialize():
        print("ReIDExtractor 初始化失败，请检查模型路径和依赖。")
        return

    # 3. 读取测试图片
    image_path = "test.jpg"  # 请替换为你自己的图片路径
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    # 4. 多模态处理流程
    print("开始多模态处理...")
    
    # 4.1 目标检测
    detections = detector.detect(image)
    print(f"检测到 {len(detections)} 个目标")
    
    # 4.2 目标跟踪
    tracks = tracker.update(detections)
    print(f"跟踪到 {len(tracks)} 条轨迹")
    
    # 4.3 姿态估计
    poses = pose_estimator.process(image)
    print(f"估计到 {len(poses)} 个人体姿态")
    
    # 4.4 特征提取（针对检测到的目标）
    print("提取目标特征...")
    
    # 为每个检测到的目标提取特征
    for i, detection in enumerate(detections):
        # 提取目标区域
        x1, y1, x2, y2 = detection.bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 确保边界框有效
        if x2 > x1 and y2 > y1:
            # 裁剪目标区域
            target_roi = image[y1:y2, x1:x2]
            
            # 使用CLIP提取视觉特征
            clip_features = clip_extractor.process(target_roi)
            print(f"目标 {i+1} (CLIP特征维度): {len(clip_features) if isinstance(clip_features, list) else clip_features.shape}")
            
            # 使用ReID提取重识别特征
            reid_features = reid_extractor.process(target_roi)
            print(f"目标 {i+1} (ReID特征维度): {len(reid_features) if isinstance(reid_features, list) else reid_features.shape}")

    # 4.5 零-shot分类（使用CLIP）
    print("进行零-shot分类...")
    
    # 定义类别标签
    class_labels = ["person", "car", "dog", "cat", "bicycle"]
    
    # 使用CLIP进行零-shot分类
    clip_scores = clip_extractor.process(image, text_prompt=", ".join(class_labels))
    print(f"零-shot分类结果: {dict(zip(class_labels, clip_scores)) if isinstance(clip_scores, list) else clip_scores}")

    # 5. 多模态结果融合与可视化
    print("融合并可视化多模态结果...")
    
    visualizer = Visualizer()
    
    # 绘制所有结果
    vis_image = visualizer.draw_results(
        image.copy(),
        detections=detections,
        tracks=tracks,
        poses=poses
    )

    # 6. 显示结果
    cv2.imshow("Multimodal Processing Results", vis_image)
    print("按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
