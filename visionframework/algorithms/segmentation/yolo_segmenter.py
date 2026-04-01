"""
YOLO11 / YOLO26 实例分割：纯 PyTorch 推理（``torch.load`` 权重 + ``YOLOSegmentHead`` / ``YOLO26SegmentHead``）。

不再依赖 ``ultralytics`` 包；官方 ``*-seg.pt`` 经 :func:`visionframework.tools.convert_ultralytics.convert_segment_weights` 映射后加载。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from visionframework.algorithms.base import BaseAlgorithm
from visionframework.core.registry import ALGORITHMS
from visionframework.core.config import resolve_config
from visionframework.core.builder import build_model
from visionframework.data.detection import Detection
from visionframework.tools.convert_ultralytics import (
    convert_segment_weights,
    load_ultralytics_pt_state_dict,
)
from visionframework.utils.nms import non_max_suppression
from visionframework.utils.yolo_instance_mask import process_mask_native

logger = logging.getLogger(__name__)

_COCO_NAMES = [
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


def _segmentation_config_root() -> Path:
    import visionframework

    pkg = Path(visionframework.__file__).resolve().parent
    return pkg.parent / "configs" / "segmentation"


def _default_yaml_for_weights(weights: str, family: str) -> Path:
    stem = Path(weights).stem.lower()
    root = _segmentation_config_root()
    if family == "yolo11":
        for prefix in ("yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x"):
            if stem.startswith(prefix):
                p = root / "yolo11" / f"{prefix}_seg.yaml"
                if p.is_file():
                    return p
        p = root / "yolo11" / "yolo11_seg.yaml"
        if p.is_file():
            return p
    elif family == "yolo26":
        for prefix in ("yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"):
            if stem.startswith(prefix):
                p = root / "yolo26" / f"{prefix}_seg.yaml"
                if p.is_file():
                    return p
        p = root / "yolo26" / "yolo26_seg.yaml"
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"无法为权重 '{weights}' 解析默认分割模型 YAML（请传入 ``model_yaml`` 指向完整 backbone+neck+head 配置）"
    )


def _resolve_weights_path(weights: str) -> str:
    p = Path(weights)
    if p.is_file():
        return str(p.resolve())
    return weights


def _model_cfg_from_yaml(model_yaml: str) -> dict:
    cfg = resolve_config(model_yaml)
    return {k: v for k, v in cfg.items() if k not in ("task", "algorithm", "postprocess")}


def _load_segment_model(
    weights: str,
    model_yaml: str,
) -> nn.Module:
    mcfg = _model_cfg_from_yaml(model_yaml)
    net = build_model(mcfg, weights=None)
    wpath = _resolve_weights_path(weights)
    ul_sd = load_ultralytics_pt_state_dict(wpath)
    vf_sd = convert_segment_weights(ul_sd)
    net.load_state_dict(vf_sd, strict=False)
    return net


class _NativeYOLONSegmenter(BaseAlgorithm):
    """Letterbox → 前向 → 解码框 + NMS → ``process_mask_native``。"""

    def __init__(
        self,
        model: nn.Module,
        *,
        input_size: Tuple[int, int] = (640, 640),
        conf: float = 0.25,
        nms_iou: float = 0.45,
        end2end: bool = False,
        class_names: Optional[List[str]] = None,
        device: str = "auto",
        fp16: bool = False,
        **_kw: Any,
    ):
        super().__init__(model=model, device=device, fp16=fp16)
        self.input_size = input_size
        self.conf = float(conf)
        self.nms_iou = float(nms_iou)
        self.end2end = bool(end2end)
        self.class_names = class_names or _COCO_NAMES

    def _preprocess(self, img: np.ndarray):
        h0, w0 = img.shape[:2]
        th, tw = self.input_size
        scale = min(th / h0, tw / w0)
        nh, nw = round(h0 * scale), round(w0 * scale)
        resized = cv2.resize(img, (nw, nh))
        pad_h, pad_w = th - nh, tw - nw
        top = round(pad_h / 2 - 0.1)
        left = round(pad_w / 2 - 0.1)
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        canvas[top : top + nh, left : left + nw] = resized
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(canvas).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)
        if self.fp16:
            tensor = tensor.half()
        return tensor, scale, (top, left), (h0, w0)

    def _decode_outputs(
        self, raw_outputs: list, scale: float, pad: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        all_boxes, all_scores = [], []
        top, left = pad

        for cls_logits, reg_raw in raw_outputs:
            B, C, H, W = cls_logits.shape
            scores = cls_logits.sigmoid()
            reg_ch = reg_raw.shape[1]
            reg_max = reg_ch // 4
            if reg_max > 1:
                reg = reg_raw.view(B, 4, reg_max, H, W).softmax(dim=2)
                proj = torch.arange(reg_max, device=reg.device, dtype=reg.dtype).view(1, 1, -1, 1, 1)
                reg = (reg * proj).sum(dim=2)
            else:
                reg = reg_raw

            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=scores.device, dtype=scores.dtype),
                torch.arange(W, device=scores.device, dtype=scores.dtype),
                indexing="ij",
            )
            stride_h = self.input_size[0] / H
            stride_w = self.input_size[1] / W
            x_c = (grid_x + 0.5) * stride_w
            y_c = (grid_y + 0.5) * stride_h
            x1 = x_c - reg[:, 0] * stride_w
            y1 = y_c - reg[:, 1] * stride_h
            x2 = x_c + reg[:, 2] * stride_w
            y2 = y_c + reg[:, 3] * stride_h
            boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(B, -1, 4)
            scores_flat = scores.permute(0, 2, 3, 1).reshape(B, -1, C)
            all_boxes.append(boxes)
            all_scores.append(scores_flat)

        boxes = torch.cat(all_boxes, dim=1)
        scores = torch.cat(all_scores, dim=1)
        boxes[..., [0, 2]] = (boxes[..., [0, 2]] - left) / scale
        boxes[..., [1, 3]] = (boxes[..., [1, 3]] - top) / scale
        return boxes[0], scores[0]

    @torch.no_grad()
    def predict(self, img: np.ndarray) -> List[Detection]:
        tensor, scale, pad, orig_hw = self._preprocess(img)
        out = self.model(tensor)
        if not isinstance(out, dict) or "det" not in out:
            raise RuntimeError("分割模型应返回 ``det`` / ``proto`` / ``mask_coeff``（YOLOSegmentHead）")
        raw = out["det"]
        proto = out["proto"]
        mask_coeff = out["mask_coeff"][0]
        h0, w0 = orig_hw

        boxes, scores = self._decode_outputs(raw, scale, pad)
        mc = mask_coeff

        if self.end2end:
            s = scores
            n, nc = s.shape
            flat = s.reshape(-1)
            k = min(flat.numel(), 300)
            confs, flat_idx = flat.topk(k, largest=True, sorted=True)
            keep = confs > self.conf
            confs = confs[keep]
            flat_idx = flat_idx[keep]
            if confs.numel() == 0:
                return []
            box_idx = (flat_idx // nc).long()
            cls_ids = (flat_idx % nc).long()
            sel_boxes = boxes[box_idx]
            mc_sel = mc[:, box_idx]
            protos = proto[0] if proto.dim() == 4 else proto
            masks_t = process_mask_native(
                protos, mc_sel.t(), sel_boxes, (h0, w0)
            )
            detections: List[Detection] = []
            for i in range(sel_boxes.shape[0]):
                cid = int(cls_ids[i].item())
                name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
                m = masks_t[i].cpu().numpy().astype(np.uint8) * 255
                bb = sel_boxes[i].cpu().numpy()
                detections.append(
                    Detection(
                        bbox=tuple(float(x) for x in bb.tolist()),
                        confidence=float(confs[i].item()),
                        class_id=cid,
                        class_name=name,
                        mask=m,
                    )
                )
            return detections

        max_scores, cls_ids = scores.max(dim=1)
        keep_mask = max_scores > self.conf
        boxes = boxes[keep_mask]
        max_scores = max_scores[keep_mask]
        cls_ids = cls_ids[keep_mask]
        mc = mc[:, keep_mask]

        if boxes.numel() == 0:
            return []

        keep_indices = non_max_suppression(
            boxes.cpu().numpy(),
            max_scores.cpu().numpy(),
            self.nms_iou,
            class_ids=cls_ids.cpu().numpy(),
        )[:300]

        boxes = boxes[keep_indices]
        max_scores = max_scores[keep_indices]
        cls_ids = cls_ids[keep_indices]
        mc = mc[:, keep_indices]

        protos = proto[0] if proto.dim() == 4 else proto
        masks_t = process_mask_native(protos, mc.t(), boxes, (h0, w0))

        detections = []
        for i in range(boxes.shape[0]):
            cid = int(cls_ids[i].item())
            name = self.class_names[cid] if cid < len(self.class_names) else str(cid)
            m = masks_t[i].cpu().numpy().astype(np.uint8) * 255
            bb = boxes[i].cpu().numpy()
            detections.append(
                Detection(
                    bbox=tuple(float(x) for x in bb.tolist()),
                    confidence=float(max_scores[i].item()),
                    class_id=cid,
                    class_name=name,
                    mask=m,
                )
            )
        return detections


@ALGORITHMS.register("YOLO11Segmenter")
class YOLO11Segmenter(_NativeYOLONSegmenter):
    """YOLO11 Segment（``yolo11n-seg.pt`` …）；纯 PyTorch，无需 ultralytics。"""

    def __init__(
        self,
        weights: Optional[str] = None,
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        fp16: bool = False,
        model: Optional[nn.Module] = None,
        model_yaml: Optional[str] = None,
        **_kw: Any,
    ):
        myaml_resolved: Optional[str] = None
        if model is None:
            if not weights:
                raise ValueError("YOLO11Segmenter 需要 ``weights`` 或预构建的 ``model``")
            myaml_resolved = model_yaml or str(_default_yaml_for_weights(str(weights), "yolo11"))
            model = _load_segment_model(str(weights), myaml_resolved)
        else:
            myaml_resolved = model_yaml
        pp: dict = resolve_config(myaml_resolved).get("postprocess") or {} if myaml_resolved else {}
        end2end = bool(pp.get("end2end", False))
        wpath = str(weights) if weights else ""
        stem = Path(wpath).stem.lower()
        if wpath and not stem.startswith("yolo11"):
            logger.warning("YOLO11Segmenter 建议使用 yolo11*-seg 权重，当前: %s", wpath)
        super().__init__(
            model,
            input_size=(imgsz, imgsz),
            conf=conf,
            nms_iou=iou,
            end2end=end2end,
            device=device,
            fp16=fp16,
        )


@ALGORITHMS.register("YOLO26Segmenter")
class YOLO26Segmenter(_NativeYOLONSegmenter):
    """YOLO26 Segment（``yolo26n-seg.pt`` …）；纯 PyTorch，无需 ultralytics。"""

    def __init__(
        self,
        weights: Optional[str] = None,
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        fp16: bool = False,
        model: Optional[nn.Module] = None,
        model_yaml: Optional[str] = None,
        **_kw: Any,
    ):
        myaml_resolved: Optional[str] = None
        if model is None:
            if not weights:
                raise ValueError("YOLO26Segmenter 需要 ``weights`` 或预构建的 ``model``")
            myaml_resolved = model_yaml or str(_default_yaml_for_weights(str(weights), "yolo26"))
            model = _load_segment_model(str(weights), myaml_resolved)
        else:
            myaml_resolved = model_yaml
        pp: dict = resolve_config(myaml_resolved).get("postprocess") or {} if myaml_resolved else {}
        end2end = bool(pp.get("end2end", True))
        wpath = str(weights) if weights else ""
        stem = Path(wpath).stem.lower()
        if wpath and not stem.startswith("yolo26"):
            logger.warning("YOLO26Segmenter 建议使用 yolo26*-seg 权重，当前: %s", wpath)
        super().__init__(
            model,
            input_size=(imgsz, imgsz),
            conf=conf,
            nms_iou=iou,
            end2end=end2end,
            device=device,
            fp16=fp16,
        )
