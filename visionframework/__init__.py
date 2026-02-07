"""
Vision Framework - A comprehensive computer vision framework

Usage:
    from visionframework import Vision

    v = Vision(model="yolov8n.pt", track=True)
    for frame, meta, result in v.run("video.mp4"):
        print(result["detections"])
"""

__version__ = "0.3.0"

# ── Primary public API ────────────────────────────────────────────────
from .api import Vision

# ── Data structures (commonly needed for type hints & tests) ──────────
from .data import Detection, Track, STrack, Pose, KeyPoint, ROI

# ── Visualization (for draw convenience) ──────────────────────────────
from .utils.visualization.unified_visualizer import Visualizer

# ── Result export (used in advanced examples) ─────────────────────────
from .utils.data.export import ResultExporter

# ── Exceptions (users may want to catch specific errors) ──────────────
from .exceptions import (
    VisionFrameworkError,
    DetectorInitializationError, DetectorInferenceError,
    TrackerInitializationError, TrackerUpdateError,
    ConfigurationError, ModelNotFoundError,
    ModelLoadError, DeviceError, DependencyError,
    DataFormatError, ProcessingError,
)

# ── Advanced: low-level classes for custom component authors ──────────
# These are for users who subclass BaseDetector etc.
# Lazy-loaded via __getattr__ to avoid import overhead for normal users.

def __getattr__(name: str):
    """Lazy-load advanced/internal symbols on demand."""
    _lazy_map = {
        # Base classes
        "BaseDetector": (".core.components.detectors.base_detector", "BaseDetector"),
        "BaseTracker": (".core.components.trackers.base_tracker", "BaseTracker"),
        "BaseProcessor": (".core.components.processors.feature_extractor", "BaseProcessor"),
        # Implementations
        "YOLODetector": (".core.components.detectors.yolo_detector", "YOLODetector"),
        "DETRDetector": (".core.components.detectors.detr_detector", "DETRDetector"),
        "RFDETRDetector": (".core.components.detectors.rfdetr_detector", "RFDETRDetector"),
        "IOUTracker": (".core.components.trackers.iou_tracker", "IOUTracker"),
        "ByteTracker": (".core.components.trackers.byte_tracker", "ByteTracker"),
        "ReIDTracker": (".core.components.trackers.reid_tracker", "ReIDTracker"),
        # Processors
        "PoseEstimator": (".core.components.processors.pose_estimator", "PoseEstimator"),
        "CLIPExtractor": (".core.components.processors.clip_extractor", "CLIPExtractor"),
        "ReIDExtractor": (".core.components.processors.reid_extractor", "ReIDExtractor"),
        # Segmenters
        "SAMSegmenter": (".core.components.segmenters.sam_segmenter", "SAMSegmenter"),
        # Pipelines
        "VisionPipeline": (".core.pipelines.pipeline", "VisionPipeline"),
        "BatchPipeline": (".core.pipelines.batch", "BatchPipeline"),
        "VideoPipeline": (".core.pipelines.video", "VideoPipeline"),
        # Plugin system
        "register_detector": (".core.plugin_system", "register_detector"),
        "register_tracker": (".core.plugin_system", "register_tracker"),
        "register_segmenter": (".core.plugin_system", "register_segmenter"),
        "register_processor": (".core.plugin_system", "register_processor"),
        "register_model": (".core.plugin_system", "register_model"),
        "register_visualizer": (".core.plugin_system", "register_visualizer"),
        "register_evaluator": (".core.plugin_system", "register_evaluator"),
        "register_custom_component": (".core.plugin_system", "register_custom_component"),
        "plugin_registry": (".core.plugin_system", "plugin_registry"),
        "model_registry": (".core.plugin_system", "model_registry"),
        # Utilities
        "Config": (".utils.io.config_models", "Config"),
        "PerformanceMonitor": (".utils.monitoring.performance", "PerformanceMonitor"),
        "Timer": (".utils.monitoring.timer", "Timer"),
        "iter_frames": (".utils.io.media_source", "iter_frames"),
        # Model tools
        "TrajectoryAnalyzer": (".utils.data.trajectory_analyzer", "TrajectoryAnalyzer"),
        "ImageAugmenter": (".utils.data_augmentation.augmenter", "ImageAugmenter"),
        "AugmentationConfig": (".utils.data_augmentation.augmenter", "AugmentationConfig"),
        "AugmentationType": (".utils.data_augmentation.augmenter", "AugmentationType"),
        "QuantizationConfig": (".utils.model_optimization.quantization", "QuantizationConfig"),
        "quantize_model": (".utils.model_optimization.quantization", "quantize_model"),
        "PruningConfig": (".utils.model_optimization.pruning", "PruningConfig"),
        "prune_model": (".utils.model_optimization.pruning", "prune_model"),
        "FineTuningConfig": (".utils.model_training.fine_tuner", "FineTuningConfig"),
        "select_model": (".utils.model_management.auto_selector", "select_model"),
        "fuse_features": (".utils.multimodal.fusion", "fuse_features"),
    }

    if name in _lazy_map:
        module_path, attr_name = _lazy_map[name]
        import importlib
        module = importlib.import_module(module_path, package=__name__)
        value = getattr(module, attr_name)
        # Cache in module namespace for next access
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Primary API
    "Vision",
    "__version__",
    # Data structures
    "Detection", "Track", "STrack", "Pose", "KeyPoint", "ROI",
    # Visualization
    "Visualizer",
    # Export
    "ResultExporter",
    # Exceptions
    "VisionFrameworkError",
    "DetectorInitializationError", "DetectorInferenceError",
    "TrackerInitializationError", "TrackerUpdateError",
    "ConfigurationError", "ModelNotFoundError",
    "ModelLoadError", "DeviceError", "DependencyError",
    "DataFormatError", "ProcessingError",
]
