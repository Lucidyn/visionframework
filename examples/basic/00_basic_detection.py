"""
00_basic_detection.py

基础目标检测示例：
- 使用 YOLODetector 对单张图片进行检测
- 代码尽量简单，便于快速上手

注意：
- 需要提前准备 YOLO 权重（如 yolov8n.pt），并放在当前工作目录或指定路径。
"""

import cv2

from visionframework import YOLODetector, Visualizer


def main() -> None:
    # 1. 创建检测器配置
    detector_config = {
        "model_path": "yolov8n.pt",  # 可替换为你的模型路径
        "device": "auto",
        "conf_threshold": 0.25,
    }

    detector = YOLODetector(detector_config)

    # 2. 初始化模型（加载权重）
    if not detector.initialize():
        print("YOLODetector 初始化失败，请检查模型路径和依赖（ultralytics、torch 等）。")
        return

    # 3. 读取测试图片
    image_path = "test.jpg"  # 请替换为你自己的图片路径
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return

    # 4. 进行检测
    detections = detector.detect(image)
    print(f"检测到 {len(detections)} 个目标")

    # 5. 使用集成的可视化 API 绘制检测结果
    visualizer = Visualizer()
    vis_image = visualizer.draw_detections(image.copy(), detections)

    # 6. 显示 / 保存结果
    cv2.imshow("Detections", vis_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

