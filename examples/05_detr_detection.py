"""
示例 05 — DETR 目标检测（Facebook 官方预训练权重）

DETR 使用 Transformer encoder-decoder 架构，无 NMS，集合预测。

前提条件:
    python tools/convert_detr.py \
        --url https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
        --output weights/detr_r50.pth --verify
"""

from pathlib import Path

import cv2
from visionframework import TaskRunner, Visualizer
from visionframework.core.config import require_detector_weights

root = Path(__file__).resolve().parent.parent
require_detector_weights(
    root,
    "runs/detection/detr/detect.yaml",
    hint="无预训练权重时 DETR 推理结果通常为空。请先运行 convert_detr 下载并转换官方权重。",
)

img = cv2.imread("test_bus.jpg")
task = TaskRunner("runs/detection/detr/detect.yaml")
result = task.process(img)

detections = result.get("detections", [])
print(f"检测到 {len(detections)} 个目标")
for det in detections:
    print(f"  类别={det.class_id}, 置信度={det.confidence:.3f}, 框={det.bbox}")

vis = Visualizer()
result_img = vis.draw_detections(img.copy(), detections)
cv2.imwrite("detr_result.jpg", result_img)
print("结果已保存至: detr_result.jpg")
