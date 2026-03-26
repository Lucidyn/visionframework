"""
示例 06 — YOLO26 端到端检测（NMS-free）

YOLO26 使用 one-to-one 检测头，无需 NMS 后处理。
支持全尺寸：n/s/m/l/x，换用 configs/detection/yolo26/yolo26s.yaml 等即可。

前提条件:
    python tools/convert_ultralytics.py --model yolo26n.pt --out weights/detection/yolo26/yolo26n_converted.pth
    （转换仅需 PyTorch，无需 pip install ultralytics）
"""

from pathlib import Path

import cv2
from visionframework import TaskRunner, Visualizer
from visionframework.core.config import require_detector_weights

root = Path(__file__).resolve().parent.parent
require_detector_weights(
    root,
    "runs/detection/yolo26/detect.yaml",
    hint="无预训练权重时推理结果通常为空，图上会像没有检测到任何目标。请先转换 yolo26n.pt。",
)

img = cv2.imread("test_bus.jpg")
task = TaskRunner("runs/detection/yolo26/detect.yaml")
result = task.process(img)

detections = result.get("detections", [])
print(f"检测到 {len(detections)} 个目标")
for det in detections:
    print(f"  类别={det.class_id}, 置信度={det.confidence:.3f}, 框={det.bbox}")

vis = Visualizer()
result_img = vis.draw_detections(img.copy(), detections)
cv2.imwrite("yolo26n_result.jpg", result_img)
print("结果已保存至: yolo26n_result.jpg")
