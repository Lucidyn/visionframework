"""
示例 03 — YOLO 实例分割（YOLO11 / YOLO26）

使用 Ultralytics Segment 权重（``yolo11n-seg.pt`` / ``yolo26n-seg.pt`` 等），
``TaskRunner`` 返回 ``{"detections": [...]}``，每项含 ``mask``。

前提::

    pip install ultralytics
    # 或: pip install -e ".[yolo-seg]"

换模型尺寸：改用 ``runs/segmentation/yolo11/yolo11s_seg.yaml`` 或
``runs/segmentation/yolo26/yolo26n_seg.yaml`` 等。

批量导出各尺寸可视化：::

    python -m visionframework.tools.save_yolo_seg_visualization
"""

from pathlib import Path

import cv2

from visionframework import TaskRunner, Visualizer

root = Path(__file__).resolve().parent.parent

# 测试图：根目录 test_bus.jpg 或 test/fixtures/bus.jpg
img_path = None
for candidate in (root / "test_bus.jpg", root / "test" / "fixtures" / "bus.jpg"):
    if candidate.is_file():
        img_path = candidate
        break
if img_path is None:
    raise SystemExit("请放置 test_bus.jpg 于仓库根目录，或使用 test/fixtures/bus.jpg")

task = TaskRunner(str(root / "runs/segmentation/yolo11/yolo11n_seg.yaml"))
img = cv2.imread(str(img_path))
if img is None:
    raise SystemExit(f"无法读取图像: {img_path}")

result = task.process(img)
dets = result.get("detections", [])
print(f"实例数: {len(dets)}")
for d in dets:
    ms = d.mask.shape if d.mask is not None else None
    print(f"  {d.class_name} conf={d.confidence:.3f} mask={ms}")

out_dir = root / "outputs" / "segmentation_demo"
out_dir.mkdir(parents=True, exist_ok=True)
vis_path = out_dir / "seg_yolo11n_demo.jpg"
cv2.imwrite(str(vis_path), Visualizer().draw(img.copy(), result))
print(f"可视化已保存: {vis_path.resolve()}")
