"""
示例 07 — RF-DETR 目标检测（Roboflow，DINOv2 backbone）

RF-DETR 使用可变形注意力 + DINOv2 backbone，直接使用 VisionFramework 原生实现运行。

注意:
    - 默认加载官方 `.pth`（需要安装 rfdetr；权重缺失会自动下载）。
"""

import cv2

from visionframework import TaskRunner, Visualizer

# 加载图片
img = cv2.imread("test_bus.jpg")

# 通过 YAML 运行（默认 nano `.pth`）
task = TaskRunner("runs/detection/rfdetr/detect_nano.yaml")
result = task.process(img)
detections = result.get("detections", [])

print(f"检测到 {len(detections)} 个目标")
for det in detections:
    print(f"  类别={det.class_id}, 置信度={det.confidence:.3f}, 框={det.bbox}")

# 可视化并保存
vis = Visualizer()
result_img = vis.draw_detections(img.copy(), detections)
cv2.imwrite("rfdetr_result.jpg", result_img)
print("结果已保存至: rfdetr_result.jpg")
