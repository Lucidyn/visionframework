"""
示例 03 — 语义分割

通过 YAML 配置文件驱动 ResNet50 分割模型。
"""

import numpy as np
from visionframework import TaskRunner

img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

task = TaskRunner("configs/runtime/segmentation.yaml")
result = task.process(img)

seg_map = result.get("segmentation")
if seg_map is not None:
    print(f"分割图尺寸: {seg_map.shape}")
    print(f"类别数: {len(np.unique(seg_map))}")
