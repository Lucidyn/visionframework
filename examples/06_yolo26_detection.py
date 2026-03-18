"""
示例 06 — YOLO26 端到端检测（NMS-free）

YOLO26 使用 one-to-one 检测头，无需 NMS 后处理。
支持全尺寸：n/s/m/l/x，换用 configs/detection/yolo26/yolo26s.yaml 等即可。

前提条件:
    pip install ultralytics
    python tools/convert_ultralytics.py --model yolo26n.pt --out weights/detection/yolo26/yolo26n_converted.pth
"""

import cv2
from visionframework import TaskRunner, Visualizer

# 加载图片
img = cv2.imread("test_bus.jpg")

# 通过 YAML 配置文件启动检测（weights 字段在 detect.yaml 中指定）
task = TaskRunner("runs/detection/yolo26/detect.yaml")
result = task.process(img)

detections = result.get("detections", [])
print(f"检测到 {len(detections)} 个目标")
for det in detections:
    print(f"  类别={det.class_id}, 置信度={det.confidence:.3f}, 框={det.bbox}")

# 可视化并保存
vis = Visualizer()
result_img = vis.draw_detections(img.copy(), detections)
cv2.imwrite("yolo26n_result.jpg", result_img)
print("结果已保存至: yolo26n_result.jpg")
