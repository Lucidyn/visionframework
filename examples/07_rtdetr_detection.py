"""
示例 07 — RT-DETR 目标检测（Ultralytics 官方 rtdetr-l / rtdetr-x，HGNet，无 NMS）

与 DETR 类似，使用 ``RTDETRDetector`` + ``Visualizer`` 保存可视化结果。
模型为框架内纯 PyTorch 实现；使用官方 ``.pt`` 时遵守 Ultralytics AGPL-3.0（见 ``NOTICE``）。

前提：下载官方 ``rtdetr-l.pt`` / ``rtdetr-x.pt`` 后分别转换::

    python -m visionframework.tools.convert_ultralytics_rtdetr_hg \\
        --weights path/to/rtdetr-l.pt --variant l \\
        --out weights/detection/rtdetr/rtdetr_l_vf.pth --verify
    python -m visionframework.tools.convert_ultralytics_rtdetr_hg \\
        --weights path/to/rtdetr-x.pt --variant x \\
        --out weights/detection/rtdetr/rtdetr_x_vf.pth --verify

运行本脚本会依次使用 ``runs/detection/rtdetr/detect.yaml``（l）与
``runs/detection/rtdetr/detect_x.yaml``（x），输出
``rtdetr_result_l.jpg``、``rtdetr_result_x.jpg``；``rtdetr_result.jpg`` 与 l 结果相同以保持旧说明兼容。
"""

from __future__ import annotations

import shutil
import cv2
from pathlib import Path

from visionframework import TaskRunner, Visualizer
from visionframework.core.config import require_detector_weights

_RTDETR_HINT = (
    "无预训练权重时 RT-DETR 推理结果通常为空。请先 convert_ultralytics_rtdetr_hg 转换官方 .pt。"
)


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    variants = [
        ("runs/detection/rtdetr/detect.yaml", "rtdetr_result_l.jpg", "RT-DETR-l"),
        ("runs/detection/rtdetr/detect_x.yaml", "rtdetr_result_x.jpg", "RT-DETR-x"),
    ]
    for run_yaml, _, label in variants:
        require_detector_weights(root, run_yaml, hint=_RTDETR_HINT, label=label)

    bus = root / "test_bus.jpg"
    if not bus.is_file():
        raise FileNotFoundError(
            f"未找到 {bus}，请将测试图放在项目根目录，或使用自己的图片路径修改本示例。"
        )
    img = cv2.imread(str(bus))
    if img is None:
        raise RuntimeError(f"无法读取 {bus}")

    vis = Visualizer()
    for run_yaml, out_name, label in variants:
        task = TaskRunner(str(root / run_yaml))
        result = task.process(img)
        detections = result.get("detections", [])
        print(f"[{label}] 检测到 {len(detections)} 个目标")
        for det in detections:
            print(f"  类别={det.class_id}, 置信度={det.confidence:.3f}, 框={det.bbox}")
        out = root / out_name
        result_img = vis.draw_detections(img.copy(), detections)
        cv2.imwrite(str(out), result_img)
        print(f"[{label}] 结果已保存至: {out.resolve()}")

    shutil.copyfile(root / "rtdetr_result_l.jpg", root / "rtdetr_result.jpg")
    print(f"（兼容）rtdetr_result.jpg 已同步为 l 结果: {(root / 'rtdetr_result.jpg').resolve()}")


if __name__ == "__main__":
    main()
