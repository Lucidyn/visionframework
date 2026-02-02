"""
Vision Framework - A comprehensive computer vision framework
"""

# Version
__version__ = "0.1.0"

from .api import (
    # Core functionality
    BaseDetector, BaseTracker, VisionPipeline,
    ROIDetector, Counter,
    SAMSegmenter,
    
    # Processors
    PoseEstimator, CLIPExtractor, ReIDExtractor,
    
    # Implementations
    YOLODetector, DETRDetector, RFDETRDetector,
    IOUTracker, ByteTracker, ReIDTracker,
    
    # Data structures
    Detection, Track, STrack, Pose, KeyPoint, ROI,
    
    # Utilities
    Visualizer, Config, ResultExporter,
    PerformanceMonitor, Timer,
    VideoProcessor, VideoWriter, process_video,
    
    # Plugin system
    plugin_registry, model_registry,
    register_detector, register_tracker,
    register_segmenter, register_model,
    register_processor, register_visualizer,
    register_evaluator, register_custom_component,
    
    # Exceptions
    VisionFrameworkError,
    DetectorInitializationError, DetectorInferenceError,
    TrackerInitializationError, TrackerUpdateError,
    ConfigurationError, ModelNotFoundError,
    ModelLoadError, DeviceError, DependencyError,
    DataFormatError, ProcessingError,
    
    # Simplified API functions
    create_detector, create_pipeline,
    process_image, process_video, create_visualizer,
    
    # Pipeline classes
    BasePipeline, BatchPipeline, VideoPipeline
)

# Export all symbols
__all__ = [
    # Core functionality
    "BaseDetector", "BaseTracker", "VisionPipeline",
    "ROIDetector", "Counter",
    "SAMSegmenter",
    
    # Processors
    "PoseEstimator", "CLIPExtractor", "ReIDExtractor",
    
    # Implementations
    "YOLODetector", "DETRDetector", "RFDETRDetector",
    "IOUTracker", "ByteTracker", "ReIDTracker",
    
    # Data structures
    "Detection", "Track", "STrack", "Pose", "KeyPoint", "ROI",
    
    # Utilities
    "Visualizer", "Config", "ResultExporter",
    "PerformanceMonitor", "Timer",
    "VideoProcessor", "VideoWriter", "process_video",
    
    # Plugin system
    "plugin_registry", "model_registry",
    "register_detector", "register_tracker",
    "register_segmenter", "register_model",
    "register_processor", "register_visualizer",
    "register_evaluator", "register_custom_component",
    
    # Exceptions
    "VisionFrameworkError",
    "DetectorInitializationError", "DetectorInferenceError",
    "TrackerInitializationError", "TrackerUpdateError",
    "ConfigurationError", "ModelNotFoundError",
    "ModelLoadError", "DeviceError", "DependencyError",
    "DataFormatError", "ProcessingError",
    
    # Simplified API functions
    "create_detector", "create_pipeline",
    "process_image", "process_video",
    "create_visualizer",
    
    # Pipeline classes
    "BasePipeline", "VisionPipeline", "BatchPipeline", "VideoPipeline",
    
    # Version
    "__version__"
]
