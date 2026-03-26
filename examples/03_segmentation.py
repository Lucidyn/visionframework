"""
示例 03 — YOLO 实例分割（YOLO11 / YOLO26）

使用 Ultralytics Segment 权重（``yolo11n-seg.pt`` / ``yolo26n-seg.pt`` 等），
输出为带 ``mask`` 的 Detection 列表。

前提:
    pip install ultralytics
"""

from pathlib import Path

import cv2

from visionframework import TaskRunner

root = Path(__file__).resolve().parent.parent
# 与检测示例一致：使用 yolo11n-seg；换尺寸请改 runs/segmentation/yolo11/yolo11s_seg.yaml 等
task = TaskRunner(str(root / "runs/segmentation/yolo11/yolo11n_seg.yaml"))
img = cv2.imread(str(root / "test_bus.jpg"))
if img is None:
    raise SystemExit("请放置 test_bus.jpg 于仓库根目录")

result = task.process(img)
dets = result.get("detections", [])
print(f"实例数: {len(dets)}")
for d in dets:
    print(f"  {d.class_name} conf={d.confidence:.3f} mask={d.mask.shape if d.mask is not None else None}")
