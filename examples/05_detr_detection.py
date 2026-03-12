"""
示例 05 — DETR 目标检测（Facebook 官方预训练权重）

DETR 使用 Transformer encoder-decoder 架构，无 NMS，集合预测。

前提条件:
    python tools/convert_detr.py \
        --url https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
        --output weights/detr_r50.pth --verify
"""

import cv2
from visionframework import TaskRunner, Visualizer

# 加载图片
img = cv2.imread("test_bus.jpg")

# 通过 YAML 配置文件启动检测（weights 字段在 detect_detr.yaml 中指定）
task = TaskRunner("configs/runtime/detect_detr.yaml")
result = task.process(img)

detections = result.get("detections", [])
print(f"检测到 {len(detections)} 个目标")
for det in detections:
    print(f"  类别={det.class_id}, 置信度={det.confidence:.3f}, 框={det.bbox}")

# 可视化并保存
vis = Visualizer()
result_img = vis.draw_detections(img.copy(), detections)
cv2.imwrite("detr_result.jpg", result_img)
print("结果已保存至: detr_result.jpg")
