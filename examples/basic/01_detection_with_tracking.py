"""
01_detection_with_tracking.py

带跟踪的目标检测示例：
- 使用 VisionPipeline 同时完成检测和多目标跟踪

注意：
- 需要 YOLO 权重（如 yolov8n.pt）
- 默认使用 ByteTrack 作为跟踪器
"""

import cv2

from visionframework import VisionPipeline, Visualizer


def main() -> None:
    # 1. 创建带跟踪的管道配置
    config = {
        "enable_tracking": True,
        "detector_config": {
            "model_path": "yolov8n.pt",
            "device": "auto",
            "conf_threshold": 0.25,
        },
        "tracker_config": {
            "tracker_type": "bytetrack",
            "max_age": 30,
        },
    }

    pipeline = VisionPipeline(config)

    # 2. 读取测试图片
    image_path = "test.jpg"  # 请替换为你自己的图片路径
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"无法读取图像: {image_path}")
        return

    # 3. 处理图像
    results = pipeline.process(frame)
    detections = results.get("detections", [])
    tracks = results.get("tracks", [])

    print(f"检测到 {len(detections)} 个目标, {len(tracks)} 条轨迹")

    # 4. 使用集成的可视化 API 绘制检测和轨迹
    visualizer = Visualizer()
    vis_frame = visualizer.draw_results(
        frame.copy(),
        detections=detections,
        tracks=tracks,
        poses=None,
    )

    # 5. 显示结果
    cv2.imshow("Detection with Tracking", vis_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

