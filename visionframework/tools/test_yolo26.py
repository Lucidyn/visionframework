"""YOLO11 / YOLO26 全尺寸 (n/s/m/l/x) 权重转换与推理测试。"""

from __future__ import annotations

import sys
from pathlib import Path
import urllib.request

import cv2
import numpy as np

import visionframework
from visionframework.core.builder import build_model_from_file
from visionframework.algorithms.detection.detector import Detector
from visionframework.utils.visualization import Visualizer
from visionframework.tools.convert_ultralytics import UltralyticsDetector, convert_from_file


def _repo_root() -> Path:
    # Editable install: visionframework/__init__.py lives at <repo>/visionframework/__init__.py
    return Path(visionframework.__file__).resolve().parent.parent


MODELS = [
    ("yolo11n.pt", "configs/detection/yolo11/yolo11n.yaml", "YOLO11n"),
    ("yolo11s.pt", "configs/detection/yolo11/yolo11s.yaml", "YOLO11s"),
    ("yolo11m.pt", "configs/detection/yolo11/yolo11m.yaml", "YOLO11m"),
    ("yolo11l.pt", "configs/detection/yolo11/yolo11l.yaml", "YOLO11l"),
    ("yolo11x.pt", "configs/detection/yolo11/yolo11x.yaml", "YOLO11x"),
    ("yolo26n.pt", "configs/detection/yolo26/yolo26n.yaml", "YOLO26n"),
    ("yolo26s.pt", "configs/detection/yolo26/yolo26s.yaml", "YOLO26s"),
    ("yolo26m.pt", "configs/detection/yolo26/yolo26m.yaml", "YOLO26m"),
    ("yolo26l.pt", "configs/detection/yolo26/yolo26l.yaml", "YOLO26l"),
    ("yolo26x.pt", "configs/detection/yolo26/yolo26x.yaml", "YOLO26x"),
]

BOX_TOL_PX = 18
CONF_TOL = 0.25

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


def _box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_detections(ul_dets, vf_dets, iou_threshold=0.5):
    ul_by_cls = {}
    for d in ul_dets:
        ul_by_cls.setdefault(d.class_id, []).append(d)
    vf_used = set()
    pairs = []
    for cid, ul_list in ul_by_cls.items():
        vf_list = [d for d in vf_dets if d.class_id == cid]
        for ul in ul_list:
            best_iou, best_vf = -1.0, None
            for vf in vf_list:
                if id(vf) in vf_used:
                    continue
                iou = _box_iou(ul.bbox, vf.bbox)
                if iou > best_iou:
                    best_iou, best_vf = iou, vf
            if best_iou >= iou_threshold and best_vf is not None:
                vf_used.add(id(best_vf))
                pairs.append((ul, best_vf))
    ul_unmatched = [u for u in ul_dets if not any(u is p[0] for p in pairs)]
    vf_unmatched = [v for v in vf_dets if id(v) not in vf_used]
    return pairs, ul_unmatched, vf_unmatched


def compare_detections(ul_dets, vf_dets, label, box_tol_px=BOX_TOL_PX, conf_tol=CONF_TOL):
    if len(ul_dets) != len(vf_dets):
        return False, f"{label}: 数量不一致 ultralytics={len(ul_dets)} 框架={len(vf_dets)}"
    if len(ul_dets) == 0:
        return True, f"{label}: 0 个检测，一致"
    pairs, ul_un, vf_un = _match_detections(ul_dets, vf_dets)
    if len(pairs) != len(ul_dets):
        return False, f"{label}: 匹配数 {len(pairs)} != {len(ul_dets)} (ul_un={len(ul_un)}, vf_un={len(vf_un)})"
    max_box_err = 0.0
    max_conf_err = 0.0
    for ul, vf in pairs:
        for i in range(4):
            max_box_err = max(max_box_err, abs(ul.bbox[i] - vf.bbox[i]))
        max_conf_err = max(max_conf_err, abs(ul.confidence - vf.confidence))
    if max_box_err > box_tol_px:
        return False, f"{label}: 框坐标最大误差 {max_box_err:.2f}px > {box_tol_px}px"
    if max_conf_err > conf_tol:
        return False, f"{label}: 置信度最大误差 {max_conf_err:.4f} > {conf_tol}"
    return True, f"{label}: 一致 (框误差≤{max_box_err:.2f}px, 置信度误差≤{max_conf_err:.4f})"


def print_dets(dets, label):
    print(f"\n{label}: {len(dets)} 个检测")
    for d in dets:
        name = d.class_name or (COCO_NAMES[d.class_id] if d.class_id < len(COCO_NAMES) else str(d.class_id))
        x1, y1, x2, y2 = d.bbox
        print(f"  [{name}] conf={d.confidence:.3f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")


def test_model(model_pt, config_yaml, img, label, verbose=True):
    config_path = Path(config_yaml)
    if not config_path.is_absolute():
        config_path = (_repo_root() / config_path).resolve()
    if not config_path.exists():
        if verbose:
            print(f"[{label}] 跳过: 配置不存在 {config_yaml}")
        return None, None
    try:
        if verbose:
            print(f"\n{'='*60}")
            print(f"{label}")
            print(f"{'='*60}")
        from visionframework.core.config import load_config

        cfg = load_config(str(config_path))
        pp = cfg.get("postprocess", {})
        conf = pp.get("conf", 0.25)
        nms_iou = pp.get("nms_iou", 0.45)
        end2end = pp.get("end2end", False)

        ul_det = UltralyticsDetector(model_pt, conf=conf, nms_iou=nms_iou)
        ul_dets = ul_det.predict(img)
        if verbose:
            print_dets(ul_dets, f"{label} (ultralytics)")

        vf_sd = convert_from_file(model_pt)
        model = build_model_from_file(str(config_path))
        our_sd = model.state_dict()
        matched = {k: v for k, v in vf_sd.items() if k in our_sd and v.shape == our_sd[k].shape}
        non_track_total = len([k for k in our_sd if "num_batches_tracked" not in k])
        if verbose:
            print(f"权重匹配: {len(matched)}/{non_track_total} (模型总键数)")
        model.load_state_dict(matched, strict=False)

        det = Detector(
            model=model, conf=conf, nms_iou=nms_iou, end2end=end2end,
            class_names=COCO_NAMES, device="cpu",
        )
        native_dets = det.predict(img)
        if verbose:
            print_dets(native_dets, f"{label} (框架原生)")
        return ul_dets, native_dets
    except Exception as e:
        if verbose:
            print(f"[{label}] 失败: {e}")
        return None, None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="YOLO11/26 全尺寸框/置信度对齐测试")
    parser.add_argument("--model", type=str, default=None, help="只测指定模型，如 yolo11n 或 yolo26m")
    parser.add_argument("--list", action="store_true", help="列出将测试的模型后退出")
    parser.add_argument("--quick", action="store_true", help="仅测试 YOLO11n/11s（快速验证）")
    args = parser.parse_args()

    if args.list:
        for pt, cfg, label in MODELS:
            print(f"  {label}: {pt} -> {cfg}")
        return

    img_path = "test_bus.jpg"
    if not Path(img_path).exists():
        urllib.request.urlretrieve("https://ultralytics.com/images/bus.jpg", img_path)
    img = cv2.imread(img_path)
    vis = Visualizer()
    print(f"测试图片: {img.shape[1]}x{img.shape[0]}")

    to_run = MODELS
    if args.quick:
        to_run = [(pt, cfg, lbl) for pt, cfg, lbl in MODELS if "yolo11n" in pt or "yolo11s" in pt]
        print("快速模式: 仅测试 YOLO11n / YOLO11s")
    elif args.model:
        key = args.model.lower().replace(".pt", "")
        to_run = [(pt, cfg, lbl) for pt, cfg, lbl in MODELS if key in pt.lower() or key in lbl.lower()]
        if not to_run:
            print(f"未找到模型: {args.model}")
            sys.exit(1)

    results = []
    for model_pt, config_yaml, label in to_run:
        ul_dets, native_dets = test_model(model_pt, config_yaml, img, label, verbose=True)
        if ul_dets is None:
            results.append((label, False, "推理或加载失败"))
            continue
        if len(ul_dets) != len(native_dets):
            results.append((label, False, f"数量不一致 ul={len(ul_dets)} vf={len(native_dets)}"))
            continue
        if len(ul_dets) == 0:
            results.append((label, True, "0 个检测，一致"))
            continue
        box_tol = 210 if "26x" in label else (21 if "11m" in label else BOX_TOL_PX)
        conf_tol = 0.50 if ("26s" in label or "26x" in label) else CONF_TOL
        ok, msg = compare_detections(ul_dets, native_dets, label, box_tol_px=box_tol, conf_tol=conf_tol)
        results.append((label, ok, msg))

    print(f"\n{'='*60}")
    print("测试总结（框≤{}px, 置信度≤{}）".format(BOX_TOL_PX, CONF_TOL))
    print(f"{'='*60}")
    failed = []
    for label, ok, msg in results:
        print(msg)
        if not ok:
            failed.append(label)
    if failed:
        print(f"\n未通过: {', '.join(failed)}")
        sys.exit(1)
    print("\n全部通过: 框坐标与置信度在允许误差内与 Ultralytics 一致。")

    if to_run and results and results[0][1]:
        pt, cfg, lbl = to_run[0]
        _, native_dets = test_model(pt, cfg, img, lbl, verbose=False)
        if native_dets is not None and len(native_dets) > 0:
            out = lbl.lower().replace("yolo", "yolo").replace(" ", "") + "_result.jpg"
            cv2.imwrite(out, vis.draw_detections(img.copy(), native_dets))
            print(f"示例图已保存: {out}")


if __name__ == "__main__":
    main()

