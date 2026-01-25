"""
Utility modules for vision framework with lazy loading for heavy dependencies
"""

from .config import Config, DeviceManager, ModelCache
from .config_models import (
    BaseConfig, DetectorConfig, TrackerConfig, PerformanceConfig,
    PipelineConfig, VisualizerConfig, AutoLabelerConfig, validate_config
)
from .image_utils import ImageUtils
from .export import ResultExporter
from .performance import PerformanceMonitor, Timer
from .video_utils import VideoProcessor, VideoWriter, process_video
from .logger import setup_logger, get_logger
from .trajectory_analyzer import TrajectoryAnalyzer


def __getattr__(name):
    """Lazy load visualization and evaluation modules to avoid heavy imports at module load time"""
    if name == "Visualizer":
        from .visualization import Visualizer
        return Visualizer
    elif name == "DetectionEvaluator":
        from .evaluation import DetectionEvaluator
        return DetectionEvaluator
    elif name == "TrackingEvaluator":
        from .evaluation import TrackingEvaluator
        return TrackingEvaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Visualizer",
    "Config",
    "DeviceManager",
    "ModelCache",
    "ImageUtils",
    "ResultExporter",
    "PerformanceMonitor",
    "Timer",
    "VideoProcessor",
    "VideoWriter",
    "process_video",
    "setup_logger",
    "get_logger",
    "TrajectoryAnalyzer",
    "DetectionEvaluator",
    "TrackingEvaluator",
    "BaseConfig",
    "DetectorConfig",
    "TrackerConfig",
    "PerformanceConfig",
    "PipelineConfig",
    "VisualizerConfig",
    "AutoLabelerConfig",
    "validate_config",
]

