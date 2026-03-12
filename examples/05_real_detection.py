"""
示例 05 — 使用 ultralytics 预训练权重进行真实检测

本示例展示如何：
1. 通过 tools/convert_ultralytics.py 转换 ultralytics 权重
2. 在 YAML 配置中指定转换后的权重路径
3. 使用 TaskRunner 进行真实图片检测

前提条件:
    pip install ultralytics
    python tools/convert_ultralytics.py --model yolo11n.pt --out weights/yolo11n_vf.pt
"""

import sys
from pathlib import Path

# 如果已有转换后的权重，可直接修改 YAML 中的 weights 字段使用
# 这里演示使用 UltralyticsDetector（无需转换权重，直接调用 ultralytics 推理）

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from tools.convert_ultralytics import UltralyticsDetector
    from visionframework import Visualizer

    import cv2
    import numpy as np

    # 下载测试图片
    img_path = "test_bus.jpg"
    if not Path(img_path).exists():
        import urllib.request
        url = "https://ultralytics.com/images/bus.jpg"
        print(f"下载测试图片: {url}")
        urllib.request.urlretrieve(url, img_path)

    img = cv2.imread(img_path)
    print(f"图片: {img_path} ({img.shape[1]}x{img.shape[0]})")

    # 使用 ultralytics 推理
    detector = UltralyticsDetector("yolo11n.pt", conf=0.25)
    detections = detector.predict(img)

    print(f"\n检测到 {len(detections)} 个目标:")
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        print(f"  [{det.class_name}] conf={det.confidence:.3f} "
              f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    # 可视化
    vis = Visualizer()
    drawn = vis.draw_detections(img.copy(), detections)
    cv2.imwrite("detection_result.jpg", drawn)
    print(f"\n结果已保存至: detection_result.jpg")

except ImportError as e:
    print(f"需要安装 ultralytics: pip install ultralytics\n{e}")
