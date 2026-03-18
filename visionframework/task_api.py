"""
统一任务接口 — 框架唯一公共入口。

用法
----
::

    from visionframework import TaskRunner

    # 图片检测
    task = TaskRunner("runs/detection/yolo11/detect.yaml")
    result = task.process(image)

    # 视频跟踪
    task = TaskRunner("runs/tracking/bytetrack/tracking.yaml")
    for frame, meta, result in task.run("data/video.mp4"):
        print(result["tracks"])
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from visionframework.core.config import load_config, resolve_config
from visionframework.core.builder import build_model, build_pipeline
from visionframework.engine.runner import Runner

import visionframework.models  # noqa: F401


def _resolve_weights(runtime_cfg: Dict[str, Any], role: str) -> Any:
    """从 runtime 配置中解析指定角色的权重路径。

    支持两种写法：
    - ``weights: path/to/weights.pth``          — 单一权重，作用于 detector
    - ``weights: {detector: ..., reid: ...}``   — 按角色分别指定
    """
    weights = runtime_cfg.get("weights")
    if weights is None:
        return None
    if isinstance(weights, dict):
        return weights.get(role)
    # 字符串形式只作用于 detector（最常见场景）
    return weights if role == "detector" else None


def _build_pipeline_from_runtime(runtime_cfg: Dict[str, Any]):
    """根据 runtime 配置组装 pipeline。"""
    from visionframework.core.registry import PIPELINES
    from visionframework.algorithms.detection.detector import Detector
    from visionframework.algorithms.segmentation.segmenter import Segmenter
    from visionframework.algorithms.reid.embedder import Embedder
    from visionframework.algorithms.tracking.byte_tracker import ByteTracker
    from visionframework.algorithms.tracking.iou_tracker import IOUTracker

    pipeline_type = runtime_cfg.get("pipeline", "detection")
    models_cfg = runtime_cfg.get("models") or runtime_cfg.get("model", {})
    device = runtime_cfg.get("device", "auto")
    fp16 = runtime_cfg.get("fp16", False)
    algorithm_type = runtime_cfg.get("algorithm", None)
    filter_classes = runtime_cfg.get("filter_classes")

    if pipeline_type == "detection":
        if algorithm_type == "RFDETRPTHDetector":
            from visionframework.algorithms.detection.rfdetr_pth_detector import RFDETRPTHDetector
            detector = RFDETRPTHDetector(
                model_size=runtime_cfg.get("model_size", "nano"),
                weights=_resolve_weights(runtime_cfg, "detector") or runtime_cfg.get("weights", "rf-detr-nano.pth"),
                resolution=runtime_cfg.get("resolution"),
                conf=runtime_cfg.get("conf", 0.5),
                num_select=runtime_cfg.get("num_select", 300),
                class_names=runtime_cfg.get("class_names"),
                filter_classes=filter_classes,
                device=device,
                fp16=fp16,
                auto_download=runtime_cfg.get("auto_download", True),
                weights_dir=runtime_cfg.get("weights_dir", "weights"),
            )
            return PIPELINES.get("detection")(detector=detector)

        det_cfg_path = models_cfg if isinstance(models_cfg, str) else models_cfg.get("detector")
        model_cfg = resolve_config(det_cfg_path) if det_cfg_path else {}
        model = build_model(model_cfg, weights=_resolve_weights(runtime_cfg, "detector"))
        pp = model_cfg.get("postprocess", {})

        if algorithm_type == "DETRDetector":
            from visionframework.algorithms.detection.detr_detector import DETRDetector
            detector = DETRDetector(
                model=model, device=device, fp16=fp16,
                conf=pp.get("conf", 0.25),
                class_names=model_cfg.get("class_names"),
                filter_classes=filter_classes,
            )
        else:
            detector = Detector(
                model=model, device=device, fp16=fp16,
                conf=pp.get("conf", 0.25), nms_iou=pp.get("nms_iou", 0.45),
                class_names=model_cfg.get("class_names"),
                filter_classes=filter_classes,
                end2end=pp.get("end2end", False),
            )
        return PIPELINES.get("detection")(detector=detector)

    if pipeline_type == "segmentation":
        seg_cfg_path = models_cfg if isinstance(models_cfg, str) else models_cfg.get("segmenter")
        model_cfg = resolve_config(seg_cfg_path) if seg_cfg_path else {}
        model = build_model(model_cfg, weights=_resolve_weights(runtime_cfg, "segmenter"))
        segmenter = Segmenter(model=model, device=device, fp16=fp16,
                              num_classes=model_cfg.get("head", {}).get("num_classes", 21))
        return PIPELINES.get("segmentation")(segmenter=segmenter)

    if pipeline_type in ("tracking", "reid_tracking"):
        det_cfg_path = models_cfg.get("detector") if isinstance(models_cfg, dict) else models_cfg
        det_model_cfg = resolve_config(det_cfg_path) if det_cfg_path else {}
        det_model = build_model(det_model_cfg, weights=_resolve_weights(runtime_cfg, "detector"))
        pp = det_model_cfg.get("postprocess", {})
        detector = Detector(
            model=det_model, device=device, fp16=fp16,
            conf=pp.get("conf", 0.25), nms_iou=pp.get("nms_iou", 0.45),
            class_names=det_model_cfg.get("class_names"),
            filter_classes=filter_classes,
            end2end=pp.get("end2end", False),
        )

        tracker_cfg = runtime_cfg.get("tracker", {})
        if isinstance(tracker_cfg, str):
            tracker_cfg = load_config(tracker_cfg)
        tracker_type = tracker_cfg.pop("type", "bytetrack").lower()
        tracker = ByteTracker(**tracker_cfg) if tracker_type == "bytetrack" else IOUTracker(**tracker_cfg)

        if pipeline_type == "reid_tracking":
            reid_cfg_path = models_cfg.get("reid")
            reid_model_cfg = resolve_config(reid_cfg_path) if reid_cfg_path else {}
            reid_model = build_model(reid_model_cfg, weights=_resolve_weights(runtime_cfg, "reid"))
            embedder = Embedder(model=reid_model, device=device, fp16=fp16)
            return PIPELINES.get("reid_tracking")(
                detector=detector, embedder=embedder, tracker=tracker,
            )

        return PIPELINES.get("tracking")(detector=detector, tracker=tracker)

    raise ValueError(f"未知的 pipeline 类型: {pipeline_type}")


class TaskRunner:
    """统一任务运行器 — 框架唯一公共入口。

    Parameters
    ----------
    yaml_path : str
        runtime YAML 配置文件路径。
    """

    def __init__(self, yaml_path: str):
        if not isinstance(yaml_path, str):
            raise TypeError(
                "TaskRunner 仅接受 YAML 配置文件路径 (str)。"
                f"传入了 {type(yaml_path).__name__}。"
            )
        self.cfg = resolve_config(yaml_path)
        self.pipeline = _build_pipeline_from_runtime(self.cfg)
        self._runner = Runner(self.pipeline)

    def run(self, source) -> Iterator[Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]]:
        """对 *source*（图片/视频/文件夹）逐帧处理，yield ``(frame, meta, result)``。"""
        yield from self._runner.run(source)

    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """处理单张图片，返回结果字典。"""
        return self.pipeline.process(image)

    def reset(self):
        """重置有状态组件（如跟踪器）。"""
        self.pipeline.reset()
