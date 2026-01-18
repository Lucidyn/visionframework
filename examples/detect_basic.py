"""
detect_basic.py
Minimal single-image detection example with parameter demonstrations.

Usage:
    python examples/detect_basic.py path/to/image.jpg

This file demonstrates common detector parameters you can configure:
    - `model_path`: 模型文件路径（例如 'yolov8n.pt')
    - `conf_threshold`: 置信度阈值，过滤低置信度检测
    - `device`: 推理设备，例如 'cpu' 或 'cuda'
    - `categories`: 可选列表，仅保留指定类别（名称或 id）

The example prints `detector.get_model_info()` and shows how to pass
`categories` to `detect()`.
"""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parents[1]))
from visionframework import Detector, Visualizer


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if img_path and Path(img_path).exists():
        image = cv2.imread(img_path)
    else:
        import numpy as np
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[:] = (128, 128, 128)

    # Initialize detector (YOLO by default). You can change options here.
    detector_cfg = {
        "model_path": "yolov8n.pt",
        "conf_threshold": 0.25,
        "device": "cpu",
        # Optional: only return detections for these classes (name or id)
        # "categories": ["person", "car"],
    }
    detector = Detector(detector_cfg)
    detector.initialize()

    # Show model info when available
    try:
        info = detector.get_model_info()
        print("Detector info:", info)
    except Exception:
        pass

    # Run detection. Example: pass categories to filter by class name or id.
    detections = detector.detect(image, categories=detector_cfg.get("categories"))
    print(f"Found {len(detections)} detections")
    for d in detections:
        print(f" - {d.class_name} (id={d.class_id}) conf={d.confidence:.2f}")

    # Visualize
    vis = Visualizer()
    out = vis.draw_detections(image, detections)
    cv2.imwrite("output_detect_basic.jpg", out)
    print("Saved visualization to output_detect_basic.jpg")


if __name__ == "__main__":
    main()
