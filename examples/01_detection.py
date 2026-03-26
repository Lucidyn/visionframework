"""
示例 01 — YOLO11 目标检测

通过 YAML 配置文件驱动 YOLO11n 检测器，处理单张图片并输出可视化结果。

前提条件:
    python tools/convert_ultralytics.py --model yolo11n.pt --out weights/detection/yolo11/yolo11n_converted.pth
    （转换仅需 PyTorch，无需 pip install ultralytics）
"""

from pathlib import Path

import cv2
from visionframework import TaskRunner, Visualizer
from visionframework.core.config import require_detector_weights

root = Path(__file__).resolve().parent.parent
require_detector_weights(
    root,
    "runs/detection/yolo11/detect.yaml",
    hint=(
        "说明：detect.yaml 里配置了 weights，但该路径不存在时框架不会报错，只会用随机初始化的网络推理，"
        "置信度极低，检测结果通常为空，保存的 yolo11n_result.jpg 上就像「什么也没检测到」。\n"
        "请先按本文件注释将官方 yolo11n.pt 转为框架权重后再运行。"
    ),
)

img = cv2.imread("test_bus.jpg")
task = TaskRunner("runs/detection/yolo11/detect.yaml")
result = task.process(img)

detections = result.get("detections", [])
print(f"检测到 {len(detections)} 个目标")
for det in detections:
    print(f"  类别={det.class_id}, 置信度={det.confidence:.3f}, 框={det.bbox}")

vis = Visualizer()
result_img = vis.draw_detections(img.copy(), detections)
cv2.imwrite("yolo11n_result.jpg", result_img)
print("结果已保存至: yolo11n_result.jpg")
