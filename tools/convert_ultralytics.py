"""
ultralytics → VisionFramework 权重转换工具。

将 ultralytics YOLO11/YOLO26 的 .pt 权重转为本框架可直接加载的 state_dict。

用法:
    python tools/convert_ultralytics.py --model yolo11n.pt --out weights/yolo11n_vf.pt
    python tools/convert_ultralytics.py --model yolo11n.pt --image test_bus.jpg --test
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visionframework.data.detection import Detection


# ── ultralytics YOLO11n/s/m/l 结构索引 → VisionFramework 键名映射 ─────────

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


def _map_conv_key(ul_suffix: str) -> str:
    """将 ultralytics Conv 内部 key 映射为 ConvBNAct 的 key。

    ultralytics Conv:  conv.weight, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.num_batches_tracked
    VisionFramework:   conv.weight, bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.num_batches_tracked

    这两者的 submodule 名一致，都是 conv + bn，不需要转换。
    """
    return ul_suffix


def _map_c3k2_key(ul_suffix: str) -> str:
    """C3k2 内部 key 映射。

    ultralytics C3k2:
      cv1.conv.weight / cv1.bn.*
      cv2.conv.weight / cv2.bn.*
      m.0.cv1.conv.weight / m.0.cv2.conv.weight ...  (Bottleneck)
      或 m.0.cv1/cv2/cv3 + m.0.m.0.cv1/cv2 (C3k -> CSPBlock 内含 Bottleneck)

    VisionFramework C3k2:
      cv1.conv.conv.weight / cv1.conv.bn.*  (外层 ConvBNAct wraps nn.Conv2d as .conv)
      cv2.conv.conv.weight / cv2.conv.bn.*
      m.0.cv1.conv.conv.weight ...

    差异：我们的 ConvBNAct 多了一层 .conv (nn.Conv2d)
    """
    return _inject_conv_wrapper(ul_suffix)


def _inject_conv_wrapper(key: str) -> str:
    """在 ConvBNAct 层加入 .conv 前缀。

    ultralytics:  cv1.conv.weight -> ours: cv1.conv.conv.weight
    ultralytics:  cv1.bn.weight   -> ours: cv1.conv.bn.weight (已匹配，ConvBNAct 用 .bn)

    实际上 ultralytics 的 Conv class 内部是 .conv (nn.Conv2d) + .bn，
    而我们的 ConvBNAct 也是 .conv (nn.Conv2d) + .bn。
    所以 key 结构一致，只需处理前缀即可。
    """
    return key


def _map_sppf_key(ul_suffix: str) -> str:
    """SPPF key 映射。"""
    return ul_suffix


def _map_c2psa_key(ul_suffix: str) -> str:
    """C2PSA 内部 key 映射。

    ultralytics C2PSA:
      cv1.conv.weight / cv1.bn.*
      cv2.conv.weight / cv2.bn.*
      m.0.attn.qkv.conv.weight / m.0.attn.qkv.bn.*
      m.0.attn.proj.conv.weight / m.0.attn.proj.bn.*
      m.0.attn.pe.conv.weight / m.0.attn.pe.bn.*
      m.0.ffn.0.conv.weight / m.0.ffn.0.bn.*
      m.0.ffn.1.conv.weight / m.0.ffn.1.bn.*

    VisionFramework C2PSA (after rewrite):
      cv1.conv.conv.weight / cv1.conv.bn.*  (ConvBNAct wraps Conv2d as .conv)
      ...same structure...
    """
    return ul_suffix


def _map_detect_key(ul_suffix: str) -> str:
    """Detect head key 映射。

    ultralytics Detect:
      cv2.{level}.{layer}.{rest}       (reg branch: Conv + Conv + Conv2d)
      cv3.{level}.{layer}.{sub}.{rest} (cls branch: Sequential(DWConv+Conv) × 2 + Conv2d)
      dfl.conv.weight                  (DFL projection)

    VisionFramework YOLOHead:
      reg_convs.{level}.{layer}.{rest}
      cls_convs.{level}.{layer}.{sub}.{rest}
      reg_preds.{level}.weight / .bias
      cls_preds.{level}.weight / .bias
    """
    # Handle reg branch (cv2)
    m = re.match(r"cv2\.(\d+)\.(\d+)\.(.*)", ul_suffix)
    if m:
        level, layer_idx, rest = m.group(1), int(m.group(2)), m.group(3)
        if layer_idx == 2:
            return f"head.reg_preds.{level}.{rest}"
        else:
            return f"head.reg_convs.{level}.{layer_idx}.{rest}"

    # Handle cls branch (cv3)
    # ultralytics structure: cv3.{level}.{layer}.{sub}.{rest}
    # layer 0/1: Sequential(DWConv, Conv1x1) — sub 0=DWConv, sub 1=Conv1x1
    # layer 2: Conv2d (final pred)
    m = re.match(r"cv3\.(\d+)\.(\d+)\.(.*)", ul_suffix)
    if m:
        level, layer_idx, rest = m.group(1), int(m.group(2)), m.group(3)
        if layer_idx == 2:
            return f"head.cls_preds.{level}.{rest}"
        else:
            return f"head.cls_convs.{level}.{layer_idx}.{rest}"

    if ul_suffix.startswith("dfl."):
        return None

    return None


def _prefix_conv(prefix: str, ul_suffix: str) -> str:
    """为 backbone/neck 中的 ConvBNAct 模块 key 添加前缀。

    ultralytics:  conv.weight -> ours: {prefix}.conv.weight
    ultralytics:  bn.weight   -> ours: {prefix}.bn.weight
    """
    return f"{prefix}.{ul_suffix}"


def _prefix_block(prefix: str, ul_suffix: str) -> str:
    """为 C3k2/SPPF/C2PSA 等模块 key 添加前缀。"""
    return f"{prefix}.{ul_suffix}"


def build_mapping(ul_state_dict: dict) -> OrderedDict:
    """构建从 ultralytics key → VisionFramework key 的完整映射。

    Returns
    -------
    OrderedDict[str, str]
        {ultralytics_key: visionframework_key}
    """
    mapping = OrderedDict()

    for ul_key in ul_state_dict.keys():
        # model.{idx}.{rest}
        m = re.match(r"model\.(\d+)\.(.*)", ul_key)
        if not m:
            continue
        idx, rest = m.group(1), m.group(2)

        vf_key = None

        if idx in BACKBONE_MAP:
            prefix = BACKBONE_MAP[idx]
            if idx in ("0", "1", "3", "5", "7"):
                vf_key = f"{prefix}.{rest}"
            elif idx == "9":
                vf_key = f"{prefix}.{rest}"
            elif idx == "10":
                vf_key = f"{prefix}.{rest}"
            else:
                vf_key = f"{prefix}.{rest}"
        elif idx in NECK_MAP:
            prefix = NECK_MAP[idx]
            if idx in ("17", "20"):
                vf_key = f"{prefix}.{rest}"
            else:
                vf_key = f"{prefix}.{rest}"
        elif idx == HEAD_IDX:
            vf_key = _map_detect_key(rest)
        else:
            continue

        if vf_key is not None:
            mapping[ul_key] = vf_key

    return mapping


def convert_weights(ul_state_dict: dict) -> OrderedDict:
    """Convert ultralytics state_dict to VisionFramework state_dict."""
    mapping = build_mapping(ul_state_dict)
    vf_state_dict = OrderedDict()
    for ul_key, vf_key in mapping.items():
        vf_state_dict[vf_key] = ul_state_dict[ul_key]
    return vf_state_dict


def convert_from_file(model_path: str, output_path: str | None = None) -> OrderedDict:
    """Load ultralytics .pt file and convert to VisionFramework state_dict.

    Parameters
    ----------
    model_path : str
        Path to ultralytics .pt file (e.g. yolo11n.pt).
    output_path : str, optional
        If given, save converted weights to this path.

    Returns
    -------
    OrderedDict
        Converted state_dict.
    """
    from ultralytics import YOLO
    yolo = YOLO(model_path)
    ul_sd = yolo.model.state_dict()

    vf_sd = convert_weights(ul_sd)
    print(f"转换完成: {len(ul_sd)} -> {len(vf_sd)} keys")

    if output_path:
        torch.save(vf_sd, output_path)
        print(f"已保存至: {output_path}")

    return vf_sd


# ── 直接使用 ultralytics 推理的 Adapter (保留兼容) ────────────────────────

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

    def __init__(self, model_path: str, conf: float = 0.25,
                 nms_iou: float = 0.45, device: str = "auto"):
        from ultralytics import YOLO
        self.conf = conf
        self.nms_iou = nms_iou
        self.yolo = YOLO(model_path)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def predict(self, img: np.ndarray) -> list:
        results = self.yolo(img, conf=self.conf, iou=self.nms_iou,
                            device=self.device, verbose=False)
        detections = []
        if results:
            r = results[0]
            for xyxy, conf_val, cls_id in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
            ):
                cid = int(cls_id)
                detections.append(Detection(
                    bbox=tuple(xyxy.tolist()),
                    confidence=float(conf_val),
                    class_id=cid,
                    class_name=self.COCO_NAMES[cid] if cid < len(self.COCO_NAMES) else str(cid),
                ))
        return detections


def main():
    parser = argparse.ArgumentParser(description="ultralytics → VisionFramework 权重转换工具")
    parser.add_argument("--model", default="yolo11n.pt", help="ultralytics 模型路径")
    parser.add_argument("--out", default=None, help="输出权重路径")
    parser.add_argument("--image", default=None, help="测试图片路径")
    parser.add_argument("--test", action="store_true", help="转换后加载到框架模型进行测试")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    out_path = args.out or str(Path(args.model).stem) + "_vf.pt"
    vf_sd = convert_from_file(args.model, out_path)

    if args.test:
        from visionframework.core.builder import build_model_from_file
        model = build_model_from_file("configs/models/yolo11n.yaml", weights=out_path)
        print(f"模型加载成功: {sum(p.numel() for p in model.parameters())} parameters")

        if args.image:
            img = cv2.imread(args.image)
            if img is None:
                print(f"错误: 无法读取图片 {args.image}")
                return
            from visionframework.algorithms.detection.detector import Detector
            det = Detector(model=model, conf=args.conf)
            results = det.predict(img)
            print(f"检测到 {len(results)} 个目标:")
            for d in results:
                x1, y1, x2, y2 = d.bbox
                print(f"  [{d.class_name or d.class_id}] conf={d.confidence:.3f} "
                      f"box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")


if __name__ == "__main__":
    main()
