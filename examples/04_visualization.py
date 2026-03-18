"""
示例 04 — 可视化工具

使用 Visualizer 在图片上绘制检测结果。
"""

import numpy as np
from visionframework import TaskRunner, Visualizer, Detection

# 创建测试图片
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# 检测
task = TaskRunner("runs/detection/yolo11/detect.yaml")
result = task.process(img)
detections = result.get("detections", [])

# 可视化
vis = Visualizer()
drawn = vis.draw_detections(img.copy(), detections)
print(f"可视化图片尺寸: {drawn.shape}")

# 也可以手动构造 Detection 进行可视化
manual_dets = [
    Detection(bbox=(50, 50, 200, 200), confidence=0.95, class_id=0, class_name="person"),
    Detection(bbox=(300, 100, 500, 300), confidence=0.8, class_id=2, class_name="car"),
]
drawn2 = vis.draw_detections(img.copy(), manual_dets)
print(f"手动检测可视化完成，{len(manual_dets)} 个目标")
