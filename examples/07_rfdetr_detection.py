"""
示例 07 — RF-DETR 目标检测（Roboflow，DINOv2 backbone）

RF-DETR 使用可变形注意力 + DINOv2 backbone，通过适配器调用官方 rfdetr 包推理。

前提条件:
    pip install rfdetr
"""

import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.rfdetr_adapter import RFDETRAdapter
from visionframework import Visualizer

# 加载图片
img = cv2.imread("test_bus.jpg")

# 通过适配器运行 RF-DETR 推理（自动下载权重）
adapter = RFDETRAdapter(model_size="base", conf=0.5)
detections = adapter.predict(img)

print(f"检测到 {len(detections)} 个目标")
for det in detections:
    print(f"  类别={det.class_id}, 置信度={det.confidence:.3f}, 框={det.bbox}")

# 可视化并保存
vis = Visualizer()
result_img = vis.draw_detections(img.copy(), detections)
cv2.imwrite("rfdetr_result.jpg", result_img)
print("结果已保存至: rfdetr_result.jpg")
