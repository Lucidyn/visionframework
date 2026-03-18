from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from visionframework.algorithms.base import BaseAlgorithm
from visionframework.core.registry import ALGORITHMS
from visionframework.data.detection import Detection
from visionframework.utils.filter import resolve_filter_ids


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _maybe_download_rfdetr_weights(filename: str, dst_path: str) -> str:
    """
    Download RF-DETR official weights (via `rfdetr` package) into *dst_path* if missing.
    This keeps runtime simple and makes `.pth` workflow reproducible.
    """
    dst = Path(dst_path)
    if dst.is_file():
        return str(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        from rfdetr.assets.model_weights import download_pretrain_weights
    except Exception as e:  # pragma: no cover
        raise ImportError("Downloading RF-DETR weights requires `pip install rfdetr`.") from e

    # rfdetr downloads to CWD if given a bare filename; we call it and then move to dst.
    download_pretrain_weights(filename)
    src = Path.cwd() / filename
    if not src.is_file():
        raise FileNotFoundError(f"rfdetr did not download expected file: {src}")
    src.replace(dst)
    return str(dst)


@ALGORITHMS.register("RFDETRPTHDetector")
class RFDETRPTHDetector(BaseAlgorithm):
    """
    RF-DETR inference using official `.pth` checkpoint (state_dict/checkpoint dict).

    Notes:
    - This implementation builds the exact RF-DETR architecture from the `rfdetr` package
      to guarantee key-compatibility with the official checkpoint.
    - You can later replace the `rfdetr` dependency by vendoring the model code while
      keeping this algorithm interface stable.
    """

    def __init__(
        self,
        model_size: str = "nano",
        weights: str = "rf-detr-nano.pth",
        resolution: int = 384,
        conf: float = 0.5,
        num_select: int = 300,
        class_names: Optional[List[str]] = None,
        filter_classes: Optional[List] = None,
        device: str = "auto",
        fp16: bool = False,
        auto_download: bool = True,
        weights_dir: str = "weights",
        **_kw,
    ):
        super().__init__(model=None, device=device, fp16=fp16)
        self.model_size = model_size
        self.resolution = int(resolution)
        self.conf = float(conf)
        self.num_select = int(num_select)
        self.class_names = class_names
        self._filter_ids = resolve_filter_ids(filter_classes, class_names)

        try:
            from rfdetr import RFDETRNano, RFDETRSmall, RFDETRBase, RFDETRMedium, RFDETRLarge
        except Exception as e:  # pragma: no cover
            raise ImportError("RFDETRPTHDetector requires `pip install rfdetr`.") from e

        cls_map = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "base": RFDETRBase,
            "medium": RFDETRMedium,
            "large": RFDETRLarge,
        }
        if model_size not in cls_map:
            raise ValueError(f"Unknown model_size: {model_size}")

        # Build wrapper and underlying torch model.
        self._wrapper = cls_map[model_size]()
        self._wrapper.model.model.eval()

        # Resolve / download checkpoint path.
        weights_path = weights
        if auto_download and not os.path.isfile(weights_path):
            weights_path = _maybe_download_rfdetr_weights(weights, str(Path(weights_dir) / weights))

        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        self._wrapper.model.model.load_state_dict(state, strict=True)

        # Ensure postprocess matches configuration.
        from rfdetr.models.lwdetr import PostProcess

        self._post = PostProcess(num_select=self.num_select)

    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h0, w0 = img_bgr.shape[:2]
        resized = cv2.resize(img_bgr, (self.resolution, self.resolution))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.fp16:
            x = x.half()
        return x, (h0, w0)

    @torch.no_grad()
    def predict(self, img: np.ndarray) -> List[Detection]:
        x, (h0, w0) = self._preprocess(img)
        outputs = self._wrapper.model.model(x)
        target_sizes = torch.tensor([(h0, w0)], device=outputs["pred_boxes"].device)
        results = self._post(outputs, target_sizes=target_sizes)[0]

        scores = results["scores"]
        labels = results["labels"]
        boxes = results["boxes"]

        keep = scores > self.conf
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        if self._filter_ids is not None and len(scores) > 0:
            mask = torch.tensor([int(c) in self._filter_ids for c in labels], dtype=torch.bool, device=labels.device)
            scores = scores[mask]
            labels = labels[mask]
            boxes = boxes[mask]

        dets: List[Detection] = []
        for i in range(len(scores)):
            cid = int(labels[i])
            x1, y1, x2, y2 = boxes[i].tolist()
            dets.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(scores[i]),
                    class_id=cid,
                    class_name=self.class_names[cid] if self.class_names and cid < len(self.class_names) else None,
                )
            )
        return dets

