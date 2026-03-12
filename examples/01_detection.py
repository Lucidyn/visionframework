"""
示例 01 — YOLO 目标检测

通过 YAML 配置文件驱动 YOLO11n 检测器，处理单张图片。
"""

import numpy as np
from visionframework import TaskRunner

# 创建随机测试图片
img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

# 通过 YAML 配置文件启动检测
task = TaskRunner("configs/runtime/detect.yaml")
result = task.process(img)

print(f"检测到 {len(result.get('detections', []))} 个目标")
for det in result.get("detections", []):
    print(f"  类别={det.class_id}, 置信度={det.confidence:.3f}, 框={det.bbox}")
