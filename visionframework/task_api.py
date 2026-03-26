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

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch.nn as nn

from visionframework.core.config import load_config, resolve_config
from visionframework.core.builder import build_model
from visionframework.core.registry import ALGORITHMS, PIPELINES
from visionframework.engine.runner import Runner
from visionframework.utils.logging_config import configure_visionframework_logging

import visionframework.models  # noqa: F401

logger = logging.getLogger(__name__)


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


def _postprocess(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    pp = model_cfg.get("postprocess")
    return pp if isinstance(pp, dict) else {}


def _build_detection_algorithm(
    model: nn.Module,
    model_cfg: Dict[str, Any],
    runtime_cfg: Dict[str, Any],
) -> Any:
    """从注册表实例化检测算法（Detector / DETRDetector / RTDETRDetector）。"""
    pp = _postprocess(model_cfg)
    device = runtime_cfg.get("device", "auto")
    fp16 = runtime_cfg.get("fp16", False)
    filter_classes = runtime_cfg.get("filter_classes")
    class_names = model_cfg.get("class_names")
    name = runtime_cfg.get("algorithm") or "Detector"

    def _kw_detector() -> Dict[str, Any]:
        return dict(
            model=model,
            device=device,
            fp16=fp16,
            conf=float(pp.get("conf", 0.25)),
            nms_iou=float(pp.get("nms_iou", 0.45)),
            class_names=class_names,
            filter_classes=filter_classes,
            end2end=bool(pp.get("end2end", False)),
        )

    def _kw_detr() -> Dict[str, Any]:
        return dict(
            model=model,
            device=device,
            fp16=fp16,
            conf=float(pp.get("conf", 0.25)),
            class_names=class_names,
            filter_classes=filter_classes,
        )

    def _kw_rtdetr() -> Dict[str, Any]:
        return dict(
            model=model,
            device=device,
            fp16=fp16,
            input_size=int(pp.get("input_size", 640)),
            conf=float(pp.get("conf", 0.5)),
            max_det=int(pp.get("max_det", 300)),
            class_names=class_names,
            filter_classes=filter_classes,
            input_layout=str(pp.get("input_layout", "bgr")),
        )

    builders = {
        "Detector": _kw_detector,
        "DETRDetector": _kw_detr,
        "RTDETRDetector": _kw_rtdetr,
    }
    if name not in builders:
        raise KeyError(
            f"Unknown detection algorithm '{name}'. Available: {list(builders.keys())}"
        )
    cls = ALGORITHMS.get(name)
    return cls(**builders[name]())


def _build_pipeline_from_runtime(
    runtime_cfg: Dict[str, Any],
    *,
    strict_weights: bool = False,
):
    """根据 runtime 配置组装 pipeline。"""
    from visionframework.algorithms.segmentation.segmenter import Segmenter
    from visionframework.algorithms.reid.embedder import Embedder
    from visionframework.algorithms.tracking.byte_tracker import ByteTracker
    from visionframework.algorithms.tracking.centroid_tracker import CentroidTracker
    from visionframework.algorithms.tracking.deepsort_tracker import DeepSortTracker
    from visionframework.algorithms.tracking.iou_tracker import IOUTracker
    from visionframework.algorithms.tracking.sort_tracker import SortTracker

    pipeline_type = runtime_cfg.get("pipeline", "detection")
    models_cfg = runtime_cfg.get("models") or runtime_cfg.get("model", {})
    device = runtime_cfg.get("device", "auto")
    fp16 = runtime_cfg.get("fp16", False)
    filter_classes = runtime_cfg.get("filter_classes")

    if pipeline_type == "detection":
        det_cfg_path = models_cfg if isinstance(models_cfg, str) else models_cfg.get("detector")
        model_cfg = resolve_config(det_cfg_path) if det_cfg_path else {}
        model = build_model(
            model_cfg,
            weights=_resolve_weights(runtime_cfg, "detector"),
            strict_weights=strict_weights,
        )
        detector = _build_detection_algorithm(model, model_cfg, runtime_cfg)
        return PIPELINES.get("detection")(detector=detector)

    if pipeline_type == "segmentation":
        seg_cfg_path = models_cfg if isinstance(models_cfg, str) else models_cfg.get("segmenter")
        model_cfg = resolve_config(seg_cfg_path) if seg_cfg_path else {}
        model = build_model(
            model_cfg,
            weights=_resolve_weights(runtime_cfg, "segmenter"),
            strict_weights=strict_weights,
        )
        segmenter = Segmenter(model=model, device=device, fp16=fp16,
                              num_classes=model_cfg.get("head", {}).get("num_classes", 21))
        return PIPELINES.get("segmentation")(segmenter=segmenter)

    if pipeline_type in ("tracking", "reid_tracking"):
        det_cfg_path = models_cfg.get("detector") if isinstance(models_cfg, dict) else models_cfg
        det_model_cfg = resolve_config(det_cfg_path) if det_cfg_path else {}
        det_model = build_model(
            det_model_cfg,
            weights=_resolve_weights(runtime_cfg, "detector"),
            strict_weights=strict_weights,
        )
        detector = _build_detection_algorithm(det_model, det_model_cfg, runtime_cfg)

        tracker_cfg = runtime_cfg.get("tracker", {})
        if isinstance(tracker_cfg, str):
            tracker_cfg = load_config(tracker_cfg)
        tracker_cfg = dict(tracker_cfg) if isinstance(tracker_cfg, dict) else {}
        tracker_type = tracker_cfg.pop("type", "bytetrack").lower()
        _tracker_builders = {
            "bytetrack": ByteTracker,
            "iou": IOUTracker,
            "centroid": CentroidTracker,
            "sort": SortTracker,
            "deepsort": DeepSortTracker,
        }
        if tracker_type not in _tracker_builders:
            raise ValueError(
                f"Unknown tracker type '{tracker_type}'. "
                f"Supported: {list(_tracker_builders.keys())}"
            )
        tracker = _tracker_builders[tracker_type](**tracker_cfg)

        if pipeline_type == "reid_tracking":
            reid_cfg_path = models_cfg.get("reid")
            reid_model_cfg = resolve_config(reid_cfg_path) if reid_cfg_path else {}
            reid_model = build_model(
                reid_model_cfg,
                weights=_resolve_weights(runtime_cfg, "reid"),
                strict_weights=strict_weights,
            )
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
    yaml_path : str | pathlib.Path
        runtime YAML 配置文件路径。
    strict_weights : bool, optional
        为 ``True`` 时，若配置中声明了权重路径但文件不存在则立即报错。
        默认读取运行配置中的 ``strict_weights``（未设置则为 ``False``）。

    日志：首次创建 ``TaskRunner`` 时会根据环境变量 ``VISIONFRAMEWORK_LOG_LEVEL``（或
    ``VF_LOG_LEVEL``）配置包级 logger；未设置时默认为 ``WARNING``（控制台不刷屏）。
    需要查看 ``TaskRunner`` 初始化说明时请设为 ``INFO``。
    """

    def __init__(
        self,
        yaml_path: Union[str, Path],
        *,
        strict_weights: Optional[bool] = None,
    ):
        configure_visionframework_logging()
        yaml_path = Path(yaml_path)
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Config not found: {yaml_path.resolve()}")
        self.cfg = resolve_config(yaml_path)
        sw = strict_weights
        if sw is None:
            sw = bool(self.cfg.get("strict_weights", False))
        self.strict_weights = sw
        algo = self.cfg.get("algorithm")
        logger.info(
            "TaskRunner pipeline=%s algorithm=%s device=%s strict_weights=%s",
            self.cfg.get("pipeline", "detection"),
            algo if algo is not None else "(default Detector)",
            self.cfg.get("device", "auto"),
            self.strict_weights,
        )
        self.pipeline = _build_pipeline_from_runtime(self.cfg, strict_weights=self.strict_weights)
        self._runner = Runner(self.pipeline)

    def run(self, source) -> Iterator[Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]]:
        """对 *source*（图片/视频/文件夹）逐帧处理，yield ``(frame, meta, result)``。"""
        yield from self._runner.run(source)

    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """处理单张图片，返回结果字典。"""
        return self.pipeline.process(image)

    def process_batch(self, images: Sequence[np.ndarray]) -> List[Dict[str, Any]]:
        """对多张已加载的 BGR 图像依次调用 :meth:`process`，顺序与输入一致。"""
        return [self.process(img) for img in images]

    def process_paths(self, paths: Sequence[Union[str, Path]]) -> List[Tuple[str, Dict[str, Any]]]:
        """按路径用 ``cv2.imread`` 读入并推理，返回 ``(绝对路径字符串, result)`` 列表。"""
        out: List[Tuple[str, Dict[str, Any]]] = []
        for p in paths:
            pp = Path(p).expanduser().resolve()
            if not pp.is_file():
                raise FileNotFoundError(f"Not a file: {pp}")
            img = cv2.imread(str(pp))
            if img is None:
                raise RuntimeError(f"Cannot decode image: {pp}")
            out.append((str(pp), self.process(img)))
        return out

    def iter_results(
        self, source: Any
    ) -> Iterator[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """与 :meth:`run` 相同数据源，但只产出 ``(meta, result)``（不保留帧拷贝）。"""
        for _frame, meta, result in self.run(source):
            yield meta, result

    def collect_results(
        self,
        source: Any,
        *,
        max_frames: Optional[int] = None,
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """将 :meth:`run` 的结果收集为列表。视频源可能极大，长视频请用 :meth:`run` 流式处理或设 ``max_frames``。"""
        rows: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        for i, (meta, result) in enumerate(self.iter_results(source)):
            if max_frames is not None and i >= max_frames:
                break
            rows.append((meta, result))
        return rows

    def reset(self):
        """重置有状态组件（如跟踪器）。"""
        self.pipeline.reset()
