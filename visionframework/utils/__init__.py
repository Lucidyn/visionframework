"""
Utility modules for vision framework with lazy loading for heavy dependencies
"""

from .io.config_models import (
    BaseConfig, DetectorConfig, TrackerConfig, PerformanceConfig,
    PipelineConfig, VisualizerConfig, AutoLabelerConfig,
    Config, DeviceManager, ModelCache
)
from .data.image_utils import ImageUtils
from .data.export import ResultExporter
from .monitoring.performance import PerformanceMonitor, Timer
from .io.video_utils import VideoProcessor, VideoWriter, process_video
from .monitoring.logger import setup_logger, get_logger
from .data.trajectory_analyzer import TrajectoryAnalyzer
from .error_handling import ErrorHandler, error_handler
from .dependency_manager import (
    DependencyManager, dependency_manager,
    is_dependency_available, get_available_dependencies,
    get_missing_dependencies, validate_dependency,
    get_install_command, import_optional_dependency, lazy_import
)


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
    "ErrorHandler",
    "error_handler",
    "DependencyManager",
    "dependency_manager",
    "is_dependency_available",
    "get_available_dependencies",
    "get_missing_dependencies",
    "validate_dependency",
    "get_install_command",
    "import_optional_dependency",
    "lazy_import",
]

