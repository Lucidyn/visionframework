"""
示例 04 — 可视化工具

在 **真实样图**（仓库根目录 ``test_bus.jpg``）上用 **手动写的示意框** 演示 ``Visualizer`` 绘制效果，
**不是**神经网络推理结果。要看真实检测请运行 ``01_detection.py`` 等，并准备好 ``runs/.../detect.yaml``
里指向的权重；若权重文件缺失，框架会用随机初始化网络推理，结果几乎总是 **0 个目标**，保存图上就像什么也没检测到。

若本地没有 ``test_bus.jpg``，会尝试从官方样例 URL 下载一次。本脚本不依赖模型权重即可完成「画框」演示。

样图来源: https://ultralytics.com/images/bus.jpg
"""

from pathlib import Path
import urllib.request

import cv2
import numpy as np

from visionframework import TaskRunner, Visualizer, Detection

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_BUS = REPO_ROOT / "test_bus.jpg"
SAMPLE_BUS_URL = "https://ultralytics.com/images/bus.jpg"

_YOLO_WEIGHTS = REPO_ROOT / "weights" / "detection" / "yolo11" / "yolo11n_converted.pth"


def _load_test_bus_bgr():
    if not TEST_BUS.is_file():
        print(f"未找到 {TEST_BUS.name}，正在下载样图…")
        TEST_BUS.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SAMPLE_BUS_URL, TEST_BUS)
    img = cv2.imread(str(TEST_BUS))
    if img is None:
        raise RuntimeError(f"无法读取 {TEST_BUS}")
    return img


# 1) 真实图片 + 手动框（公交样图上大致对齐行人/车身，仅演示绘制，加粗线便于辨认）
img = _load_test_bus_bgr()
h, w = img.shape[:2]
manual_dets = [
    Detection(bbox=(40, int(h * 0.34), 280, int(h * 0.86)), confidence=0.95, class_id=0, class_name="person"),
    Detection(bbox=(int(w * 0.30), int(h * 0.32), int(w * 0.92), int(h * 0.88)), confidence=0.88, class_id=5, class_name="bus"),
]
vis = Visualizer({"line_thickness": 4, "font_scale": 0.75})
drawn_manual = vis.draw_detections(img.copy(), manual_dets)
out_manual = REPO_ROOT / "visualization_demo.jpg"
cv2.imwrite(str(out_manual), drawn_manual)
print(
    f"[示意框，非模型输出] 已保存: {out_manual.resolve()} （{w}x{h}，{len(manual_dets)} 个手动框，线宽=4）"
)

# 2) 可选：YOLO 在真实样图上推理（需已转换权重）
if _YOLO_WEIGHTS.is_file():
    task = TaskRunner(str(REPO_ROOT / "runs" / "detection" / "yolo11" / "detect.yaml"))
    result = task.process(img)
    detections = result.get("detections", [])
    vis_yolo = Visualizer({"line_thickness": 3, "font_scale": 0.65})
    drawn_yolo = vis_yolo.draw_detections(img.copy(), detections)
    out_yolo = REPO_ROOT / "visualization_yolo11_bus.jpg"
    cv2.imwrite(str(out_yolo), drawn_yolo)
    print(f"YOLO 检测 {len(detections)} 个目标，已保存: {out_yolo.resolve()}")
else:
    print(f"跳过 YOLO 演示（未找到权重 {_YOLO_WEIGHTS}）")
