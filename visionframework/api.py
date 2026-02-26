"""
Vision Framework — Public API

Two ways to use the framework:

1. **Direct construction** (keyword arguments):

    >>> from visionframework import Vision
    >>> v = Vision(model="yolov8n.pt", track=True)
    >>> for frame, meta, result in v.run("video.mp4"):
    ...     print(result["detections"])

2. **Config file** (JSON / YAML):

    >>> from visionframework import Vision
    >>> v = Vision.from_config("config.json")
    >>> for frame, meta, result in v.run("video.mp4"):
    ...     print(result["detections"])

Everything is available via ``from visionframework import Vision``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

from .core.pipelines.pipeline import VisionPipeline
from .utils.visualization.unified_visualizer import Visualizer
from .core.roi_detector import ROIDetector
from .core.counter import Counter


class Vision:
    """One object to rule them all.

    Create a Vision instance, then call :meth:`run` on any media source.

    Examples
    --------
    **Simplest usage** — detect objects in an image:

    >>> v = Vision()
    >>> for frame, meta, result in v.run("photo.jpg"):
    ...     print(len(result["detections"]), "objects")

    **With tracking + pose estimation**:

    >>> v = Vision(model="yolov8n.pt", track=True, pose=True)
    >>> for frame, meta, result in v.run("video.mp4"):
    ...     print(result["tracks"], result["poses"])

    **From a config file** (JSON or YAML):

    >>> v = Vision.from_config("my_config.json")
    >>> for frame, meta, result in v.run(0):  # webcam
    ...     ...
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model: str = "yolov8n.pt",
        model_type: str = "yolo",
        device: str = "auto",
        conf: float = 0.25,
        iou: float = 0.45,
        track: bool = False,
        tracker: str = "bytetrack",
        segment: bool = False,
        pose: bool = False,
        fp16: bool = False,
        batch_inference: bool = False,
        dynamic_batch: bool = False,
        max_batch_size: int = 8,
        category_thresholds: Optional[Dict[str, float]] = None,
        **extra,
    ) -> None:
        """Create a Vision instance from keyword arguments.

        Parameters
        ----------
        model : str
            Path to model weights or model name (e.g. ``"yolov8n.pt"``).
        model_type : str
            Detector backend: ``"yolo"`` | ``"detr"`` | ``"rfdetr"``.
        device : str
            ``"auto"`` | ``"cpu"`` | ``"cuda"`` | ``"cuda:0"`` …
        conf : float
            Confidence threshold for detections.
        iou : float
            IoU threshold for NMS.
        track : bool
            Enable object tracking.
        tracker : str
            Tracker type: ``"bytetrack"`` | ``"ioutracker"`` | ``"reidtracker"``.
        segment : bool
            Enable instance segmentation.
        pose : bool
            Enable pose estimation.
        fp16 : bool
            Use FP16 half-precision inference (CUDA only, improves speed).
        batch_inference : bool
            Enable batch inference for processing multiple images.
        dynamic_batch : bool
            Enable dynamic batch sizing (auto-adjust batch size).
        max_batch_size : int
            Maximum batch size when batch_inference is enabled.
        category_thresholds : dict | None
            Per-category confidence thresholds, e.g. ``{"person": 0.5, "car": 0.3}``.
        **extra
            Forwarded into the detector / tracker / pipeline config as-is.
        """
        self._model = model
        self._model_type = model_type
        self._device = device
        self._conf = conf
        self._iou = iou
        self._track = track
        self._tracker = tracker
        self._segment = segment
        self._pose = pose
        self._fp16 = fp16
        self._batch_inference = batch_inference
        self._dynamic_batch = dynamic_batch
        self._max_batch_size = max_batch_size
        self._category_thresholds = category_thresholds
        self._extra = extra

        # ROI / counting (populated via add_roi)
        self._roi_detector: Optional[ROIDetector] = None
        self._counter: Optional[Counter] = None

        # Build internal pipeline
        self._pipeline = self._build_pipeline()

    # ------------------------------------------------------------------
    # Alternate constructor — from config file
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        path: Union[str, Path, dict],
    ) -> "Vision":
        """Create a Vision instance from a config file or dict.

        The config can be a **file path** (``.json`` / ``.yaml`` / ``.yml``)
        or a plain ``dict``.

        Parameters
        ----------
        path : str | Path | dict
            Path to a JSON/YAML configuration file, or a dict.

        Returns
        -------
        Vision

        Config file example (JSON)::

            {
                "model": "yolov8n.pt",
                "model_type": "yolo",
                "device": "auto",
                "conf": 0.25,
                "iou": 0.45,
                "track": true,
                "tracker": "bytetrack",
                "segment": false,
                "pose": false
            }
        """
        if isinstance(path, dict):
            cfg = path
        else:
            cfg = cls._load_config_file(path)

        return cls(**cfg)

    # ------------------------------------------------------------------
    # Core method — process any source
    # ------------------------------------------------------------------

    def run(
        self,
        source: Union[str, int, List[Union[str, int]], np.ndarray, Path],
        *,
        recursive: bool = False,
        skip_frames: int = 0,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> Iterator[Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]]:
        """Process any media source and yield results frame-by-frame.

        Parameters
        ----------
        source
            Anything you can throw at it:

            - ``"image.jpg"`` — single image file
            - ``"video.mp4"`` — video file
            - ``0`` — camera index
            - ``"rtsp://..."`` — RTSP / HTTP stream
            - ``"folder/"`` — folder of images/videos
            - ``["a.jpg", "b.mp4", 0]`` — list of mixed sources
            - ``np.ndarray`` — a single BGR image

        recursive : bool
            When *source* is a folder, recurse into sub-folders.
        skip_frames : int
            For video, skip this many frames between reads.
        start_frame : int
            For video, start at this frame index.
        end_frame : int | None
            For video, stop at this frame index (``None`` = to the end).

        Yields
        ------
        (frame, meta, result)
            - *frame*: BGR ``np.ndarray``
            - *meta*: dict with ``source_path``, ``frame_index``, …
            - *result*: dict with ``detections``, ``tracks``, ``poses``
        """
        for frame, meta, result in self._pipeline.process_source(
            source,
            recursive_folder=recursive,
            video_skip_frames=skip_frames,
            video_start_frame=start_frame,
            video_end_frame=end_frame,
        ):
            # Inject ROI counts when zones have been registered
            if self._counter is not None:
                tracks = result.get("tracks") or []
                detections = result.get("detections") or []
                if tracks:
                    result["counts"] = self._counter.count_tracks(tracks)
                else:
                    result["counts"] = self._counter.count_detections(detections)
            yield frame, meta, result

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        result: Dict[str, Any],
        **kwargs,
    ) -> np.ndarray:
        """Draw detection / tracking results on a frame.

        Parameters
        ----------
        frame : np.ndarray
            The BGR image to draw on.
        result : dict
            The *result* dict from :meth:`run`.

        Returns
        -------
        np.ndarray
            Copy of *frame* with bounding boxes, tracks, poses drawn.
        """
        vis = Visualizer(kwargs)
        return vis.draw(frame, result)

    @property
    def pipeline(self) -> VisionPipeline:
        """Access the underlying ``VisionPipeline`` for advanced usage."""
        return self._pipeline

    def cleanup(self) -> None:
        """Release model resources and free GPU memory."""
        self._pipeline.cleanup()

    # ------------------------------------------------------------------
    # ROI counting
    # ------------------------------------------------------------------

    def add_roi(
        self,
        name: str,
        points: List[Tuple[float, float]],
        roi_type: str = "polygon",
    ) -> "Vision":
        """Register a Region of Interest for counting.

        After adding at least one ROI, the ``result["counts"]`` key will be
        populated on every frame returned by :meth:`run`.

        Parameters
        ----------
        name : str
            Human-readable name for the zone (e.g. ``"entrance"``).
        points : list of (x, y)
            Polygon vertices (or two corner points for ``roi_type="rectangle"``).
        roi_type : str
            ``"polygon"`` | ``"rectangle"`` | ``"circle"``.

        Returns
        -------
        Vision
            *self*, so calls can be chained.

        Example
        -------
        >>> v = Vision(model="yolov8n.pt", track=True)
        >>> v.add_roi("entrance", [(100,100),(400,100),(400,400),(100,400)])
        >>> for frame, meta, result in v.run("video.mp4"):
        ...     print(result["counts"])
        """
        if self._roi_detector is None:
            self._roi_detector = ROIDetector()
            self._roi_detector.initialize()
            self._counter = Counter({"roi_detector": self._roi_detector})
            self._counter.initialize()
        self._roi_detector.add_roi(name, points, roi_type)
        return self

    # ------------------------------------------------------------------
    # Batch convenience
    # ------------------------------------------------------------------

    def process_batch(
        self,
        images: List[np.ndarray],
    ) -> List[Dict[str, Any]]:
        """Process a list of numpy images and return a list of result dicts.

        Parameters
        ----------
        images : list of np.ndarray
            BGR images to process.

        Returns
        -------
        list of dict
            One result dict per image (same structure as ``run()``).

        Example
        -------
        >>> v = Vision(model="yolov8n.pt")
        >>> results = v.process_batch([frame1, frame2, frame3])
        >>> for r in results:
        ...     print(len(r["detections"]))
        """
        results = []
        for img in images:
            frames = list(self.run(img))
            results.append(frames[0][2] if frames else {"detections": [], "tracks": [], "poses": []})
        return results

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """Return a summary of this Vision instance's configuration.

        Returns
        -------
        dict
            Keys: ``model``, ``model_type``, ``device``, ``conf``, ``iou``,
            ``track``, ``tracker``, ``segment``, ``pose``, ``fp16``,
            ``batch_inference``, ``max_batch_size``, ``category_thresholds``,
            ``rois``.

        Example
        -------
        >>> v = Vision(model="yolov8n.pt", track=True)
        >>> print(v.info())
        """
        rois = []
        if self._roi_detector is not None:
            rois = [r.name for r in self._roi_detector.get_rois()]
        return {
            "model": self._model,
            "model_type": self._model_type,
            "device": self._device,
            "conf": self._conf,
            "iou": self._iou,
            "track": self._track,
            "tracker": self._tracker,
            "segment": self._segment,
            "pose": self._pose,
            "fp16": self._fp16,
            "batch_inference": self._batch_inference,
            "max_batch_size": self._max_batch_size,
            "category_thresholds": self._category_thresholds,
            "rois": rois,
        }

    def __repr__(self) -> str:
        parts = [f"model={self._model!r}"]
        if self._track:
            parts.append(f"track={self._tracker!r}")
        if self._segment:
            parts.append("segment=True")
        if self._pose:
            parts.append("pose=True")
        if self._fp16:
            parts.append("fp16=True")
        if self._batch_inference:
            parts.append("batch_inference=True")
        parts.append(f"device={self._device!r}")
        return f"Vision({', '.join(parts)})"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> VisionPipeline:
        """Construct and initialise the underlying VisionPipeline."""
        detector_cfg: Dict[str, Any] = {
            "model_path": self._model,
            "model_type": self._model_type,
            "device": self._device,
            "conf_threshold": self._conf,
            "iou_threshold": self._iou,
            "use_fp16": self._fp16,
            "batch_inference": self._batch_inference,
            "dynamic_batch_size": self._dynamic_batch,
            "max_batch_size": self._max_batch_size,
        }
        if self._category_thresholds:
            detector_cfg["category_thresholds"] = self._category_thresholds

        # Segmentation flag
        if self._segment:
            detector_cfg["enable_segmentation"] = True

        # Forward any extra keys into detector config
        for k, v in self._extra.items():
            detector_cfg.setdefault(k, v)

        tracker_cfg: Dict[str, Any] = {}
        if self._track:
            tracker_cfg["tracker_type"] = self._tracker

        pipeline_cfg: Dict[str, Any] = {
            "detector_config": detector_cfg,
            "enable_tracking": self._track,
            "tracker_config": tracker_cfg,
            "enable_segmentation": self._segment,
            "enable_pose_estimation": self._pose,
        }

        pipeline = VisionPipeline(pipeline_cfg)
        pipeline.initialize()
        return pipeline

    @staticmethod
    def _load_config_file(path: Union[str, Path]) -> Dict[str, Any]:
        """Load a JSON or YAML file and return a dict."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")

        text = p.read_text(encoding="utf-8")
        suffix = p.suffix.lower()

        if suffix == ".json":
            return json.loads(text)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "PyYAML is required for YAML config files. "
                    "Install with: pip install pyyaml"
                )
            return yaml.safe_load(text)
        else:
            raise ValueError(
                f"Unsupported config format '{suffix}'. "
                f"Use .json, .yaml, or .yml"
            )


__all__ = ["Vision"]
