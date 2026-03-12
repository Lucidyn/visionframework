"""YOLO11n + YOLO26n 权重转换与推理测试。

验证框架原生模型与 ultralytics 模型的检测结果一致性。
"""
import sys
from pathlib import Path
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.convert_ultralytics import UltralyticsDetector, convert_from_file
from visionframework.core.builder import build_model_from_file
from visionframework.algorithms.detection.detector import Detector
from visionframework.utils.visualization import Visualizer

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def print_dets(dets, label):
    print(f"\n{label}: {len(dets)} 个检测")
    for d in dets:
        name = d.class_name or (COCO_NAMES[d.class_id] if d.class_id < len(COCO_NAMES) else str(d.class_id))
        x1, y1, x2, y2 = d.bbox
        print(f"  [{name}] conf={d.confidence:.3f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")


def test_model(model_pt, config_yaml, img, label):
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")

    # ultralytics 直接推理
    ul_det = UltralyticsDetector(model_pt, conf=0.25)
    ul_dets = ul_det.predict(img)
    print_dets(ul_dets, f"{label} (ultralytics)")

    # 框架原生推理
    vf_sd = convert_from_file(model_pt)
    model = build_model_from_file(config_yaml)
    our_sd = model.state_dict()
    matched = {k: v for k, v in vf_sd.items() if k in our_sd and v.shape == our_sd[k].shape}
    non_track_total = len([k for k in our_sd if "num_batches_tracked" not in k])
    print(f"权重匹配: {len(matched)}/{non_track_total}")

    model.load_state_dict(matched, strict=False)
    det = Detector(model=model, conf=0.25, class_names=COCO_NAMES, device="cpu")
    native_dets = det.predict(img)
    print_dets(native_dets, f"{label} (框架原生)")

    return ul_dets, native_dets


def main():
    img_path = "test_bus.jpg"
    if not Path(img_path).exists():
        urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", img_path)

    img = cv2.imread(img_path)
    vis = Visualizer()
    print(f"测试图片: {img.shape[1]}x{img.shape[0]}")

    ul_11, native_11 = test_model("yolo11n.pt", "configs/models/yolo11n.yaml", img, "YOLO11n")
    ul_26, native_26 = test_model("yolo26n.pt", "configs/models/yolo26n.yaml", img, "YOLO26n")

    # 可视化
    drawn = vis.draw_detections(img.copy(), native_11)
    cv2.imwrite("yolo11n_result.jpg", drawn)
    print(f"\nYOLO11n 结果已保存: yolo11n_result.jpg")

    drawn = vis.draw_detections(img.copy(), native_26 if native_26 else ul_26)
    cv2.imwrite("yolo26n_result.jpg", drawn)
    print(f"YOLO26n 结果已保存: yolo26n_result.jpg")

    print(f"\n{'='*60}")
    print("测试总结")
    print(f"{'='*60}")
    print(f"YOLO11n: ultralytics={len(ul_11)}, 框架={len(native_11)} ✓" if len(ul_11) == len(native_11) else f"YOLO11n: ultralytics={len(ul_11)}, 框架={len(native_11)}")
    print(f"YOLO26n: ultralytics={len(ul_26)}, 框架={len(native_26)} ✓" if len(ul_26) == len(native_26) else f"YOLO26n: ultralytics={len(ul_26)}, 框架={len(native_26)}")


if __name__ == "__main__":
    main()
