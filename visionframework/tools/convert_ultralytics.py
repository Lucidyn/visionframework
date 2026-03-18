"""
ultralytics → VisionFramework 权重转换工具。

将 ultralytics YOLO11/YOLO26 的 .pt 权重转为本框架可直接加载的 state_dict。

用法:
    python -m visionframework.tools.convert_ultralytics --model yolo11n.pt --out weights/detection/yolo11/yolo11n_converted.pth
    python -m visionframework.tools.convert_ultralytics --model yolo11n.pt --image test_bus.jpg --test
"""

from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from visionframework.data.detection import Detection


# ── ultralytics YOLO11n/s/m/l 结构索引 → VisionFramework 键名映射 ─────────
#
# ultralytics model.{i} 对应的模块（以 YOLO11 为例）:
#   0: Conv  -> backbone.conv0
#   1: Conv  -> backbone.conv1
#   2: C3k2  -> backbone.c3k2_1
#   3: Conv  -> backbone.conv3
#   4: C3k2  -> backbone.c3k2_2
#   5: Conv  -> backbone.conv5
#   6: C3k2  -> backbone.c3k2_3
#   7: Conv  -> backbone.conv7
#   8: C3k2  -> backbone.c3k2_4
#   9: SPPF  -> backbone.sppf
#  10: C2PSA -> backbone.c2psa
#  11: nn.Upsample (无参数)
#  12: torch.cat (无参数)
#  13: C3k2  -> neck.td_c3k2_p4
#  14: nn.Upsample (无参数)
#  15: torch.cat (无参数)
#  16: C3k2  -> neck.td_c3k2_p3
#  17: Conv  -> neck.bu_down_p3
#  18: torch.cat (无参数)
#  19: C3k2  -> neck.bu_c3k2_p4
#  20: Conv  -> neck.bu_down_p4
#  21: torch.cat (无参数)
#  22: C3k2  -> neck.bu_c3k2_p5
#  23: Detect head

BACKBONE_MAP = {
    "0": "backbone.conv0",
    "1": "backbone.conv1",
    "2": "backbone.c3k2_1",
    "3": "backbone.conv3",
    "4": "backbone.c3k2_2",
    "5": "backbone.conv5",
    "6": "backbone.c3k2_3",
    "7": "backbone.conv7",
    "8": "backbone.c3k2_4",
    "9": "backbone.sppf",
    "10": "backbone.c2psa",
}

NECK_MAP = {
    "13": "neck.td_c3k2_p4",
    "16": "neck.td_c3k2_p3",
    "17": "neck.bu_down_p3",
    "19": "neck.bu_c3k2_p4",
    "20": "neck.bu_down_p4",
    "22": "neck.bu_c3k2_p5",
}

HEAD_IDX = "23"


def _map_detect_key(ul_suffix: str, use_one2one: bool = False) -> str | None:
    """Detect head key 映射。

    ultralytics YOLO11 Detect:
      cv2.{level}.{layer}.{rest}       (reg branch: Conv + Conv + Conv2d)
      cv3.{level}.{layer}.{sub}.{rest} (cls branch: Sequential(DWConv+Conv) × 2 + Conv2d)
      dfl.conv.weight                  (DFL projection)

    ultralytics YOLO26 v2 Detect (8.4+):
      cv2/cv3          -> one-to-many head (training only, skip)
      one2one_cv2/cv3  -> one-to-one head  (inference, use this)

    VisionFramework YOLOHead:
      reg_convs.{level}.{layer}.{rest}
      cls_convs.{level}.{layer}.{sub}.{rest}
      reg_preds.{level}.weight / .bias
      cls_preds.{level}.weight / .bias
    """
    # For YOLO26 (use_one2one=True): map one2one_cv2/cv3, skip plain cv2/cv3
    if use_one2one:
        if ul_suffix.startswith("cv2.") or ul_suffix.startswith("cv3."):
            return None
        # Strip "one2one_" prefix and fall through to normal mapping
        ul_suffix = ul_suffix.replace("one2one_cv2.", "cv2.", 1).replace("one2one_cv3.", "cv3.", 1)
    else:
        # For YOLO11: skip one2one_* keys if present
        if ul_suffix.startswith("one2one_"):
            return None

    # Handle reg branch (cv2)
    m = re.match(r"cv2\.(\d+)\.(\d+)\.(.*)", ul_suffix)
    if m:
        level, layer_idx, rest = m.group(1), int(m.group(2)), m.group(3)
        if layer_idx == 2:
            return f"head.reg_preds.{level}.{rest}"
        return f"head.reg_convs.{level}.{layer_idx}.{rest}"

    # Handle cls branch (cv3)
    m = re.match(r"cv3\.(\d+)\.(\d+)\.(.*)", ul_suffix)
    if m:
        level, layer_idx, rest = m.group(1), int(m.group(2)), m.group(3)
        if layer_idx == 2:
            return f"head.cls_preds.{level}.{rest}"
        m2 = re.match(r"(\d+)\.(.*)", rest)
        if not m2:
            return None
        sub_idx, rest2 = int(m2.group(1)), m2.group(2)
        return f"head.cls_convs.{level}.{layer_idx}.{sub_idx}.{rest2}"

    if ul_suffix.startswith("dfl."):
        return None
    return None


def build_mapping(ul_state_dict: dict) -> OrderedDict[str, str]:
    """构建从 ultralytics key → VisionFramework key 的完整映射。"""
    use_one2one = any(f"model.{HEAD_IDX}.one2one_cv2" in k for k in ul_state_dict.keys())
    if use_one2one:
        print("检测到 YOLO26 (one-to-one head)，使用 one2one_cv2/cv3 权重")

    mapping: OrderedDict[str, str] = OrderedDict()
    for ul_key in ul_state_dict.keys():
        m = re.match(r"model\.(\d+)\.(.*)", ul_key)
        if not m:
            continue
        idx, rest = m.group(1), m.group(2)

        vf_key = None
        if idx in BACKBONE_MAP:
            vf_key = f"{BACKBONE_MAP[idx]}.{rest}"
        elif idx in NECK_MAP:
            vf_key = f"{NECK_MAP[idx]}.{rest}"
        elif idx == HEAD_IDX:
            vf_key = _map_detect_key(rest, use_one2one=use_one2one)

        if vf_key is not None:
            mapping[ul_key] = vf_key
    return mapping


def convert_weights(ul_state_dict: dict) -> OrderedDict[str, torch.Tensor]:
    mapping = build_mapping(ul_state_dict)
    vf_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
    for ul_key, vf_key in mapping.items():
        vf_state_dict[vf_key] = ul_state_dict[ul_key]
    return vf_state_dict


def convert_from_file(model_path: str, output_path: str | None = None) -> OrderedDict[str, torch.Tensor]:
    from ultralytics import YOLO

    yolo = YOLO(model_path)
    ul_sd = yolo.model.state_dict()
    vf_sd = convert_weights(ul_sd)
    print(f"转换完成: {len(ul_sd)} -> {len(vf_sd)} keys")

    if output_path:
        torch.save(vf_sd, output_path)
        print(f"已保存至: {output_path}")
    return vf_sd


class UltralyticsDetector:
    """直接使用 ultralytics 模型进行检测，输出本框架的 Detection 对象。"""

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

    def __init__(self, model_path: str, conf: float = 0.25, nms_iou: float = 0.45, device: str = "auto"):
        from ultralytics import YOLO

        self.conf = conf
        self.nms_iou = nms_iou
        self.yolo = YOLO(model_path)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def predict(self, img: np.ndarray) -> list[Detection]:
        results = self.yolo(img, conf=self.conf, iou=self.nms_iou, device=self.device, verbose=False)
        detections: list[Detection] = []
        if results:
            r = results[0]
            for xyxy, conf_val, cls_id in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
            ):
                cid = int(cls_id)
                detections.append(
                    Detection(
                        bbox=tuple(xyxy.tolist()),
                        confidence=float(conf_val),
                        class_id=cid,
                        class_name=self.COCO_NAMES[cid] if cid < len(self.COCO_NAMES) else str(cid),
                    )
                )
        return detections


def main() -> None:
    parser = argparse.ArgumentParser(description="ultralytics → VisionFramework 权重转换工具")
    parser.add_argument("--model", default="yolo11n.pt", help="ultralytics 模型路径")
    parser.add_argument("--out", default=None, help="输出权重路径")
    parser.add_argument("--image", default=None, help="测试图片路径")
    parser.add_argument("--test", action="store_true", help="转换后加载到框架模型进行测试")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    out_path = args.out or (str(Path(args.model).stem) + "_vf.pt")
    convert_from_file(args.model, out_path)

    if args.test:
        from visionframework.core.builder import build_model_from_file
        from visionframework.algorithms.detection.detector import Detector

        model_stem = Path(args.model).stem  # e.g. "yolo26s"
        # New config layout: configs/<task>/<algo>/<name>.yaml
        if model_stem.startswith("yolo11"):
            cfg_path = f"configs/detection/yolo11/{model_stem}.yaml"
        elif model_stem.startswith("yolo26"):
            cfg_path = f"configs/detection/yolo26/{model_stem}.yaml"
        else:
            cfg_path = f"configs/detection/yolo11/{model_stem}.yaml"
        if not Path(cfg_path).exists():
            cfg_path = "configs/detection/yolo11/yolo11n.yaml"
            print(f"警告: 未找到 {model_stem}.yaml，使用默认 yolo11n.yaml")
        model = build_model_from_file(cfg_path, weights=out_path)
        print(f"模型加载成功 ({cfg_path}): {sum(p.numel() for p in model.parameters())} parameters")

        if args.image:
            img = cv2.imread(args.image)
            if img is None:
                print(f"错误: 无法读取图片 {args.image}")
                return
            det = Detector(model=model, conf=args.conf)
            results = det.predict(img)
            print(f"检测到 {len(results)} 个目标:")
            for d in results:
                x1, y1, x2, y2 = d.bbox
                print(
                    f"  [{d.class_name or d.class_id}] conf={d.confidence:.3f} "
                    f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})"
                )


if __name__ == "__main__":
    main()

