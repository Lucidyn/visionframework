"""
Unified API entry point for Vision Framework

Provides a simplified interface to access all core functionality
without complex import paths.
"""

from typing import Optional, Dict, Any, List, Union
import numpy as np

# Core functionality, processors, implementations, and plugin system
from .core import (
    # Core classes
    BaseDetector, BaseTracker, VisionPipeline,
    ROIDetector, Counter, SAMSegmenter,
    
    # Processors
    PoseEstimator, CLIPExtractor, ReIDExtractor,
    
    # Implementations
    YOLODetector, DETRDetector, RFDETRDetector,
    IOUTracker, ByteTracker, ReIDTracker,
    
    # Plugin system
    plugin_registry, model_registry,
    register_detector, register_tracker,
    register_segmenter, register_model,
    register_processor, register_visualizer,
    register_evaluator, register_custom_component,
    
    # Pipeline classes
    BasePipeline, BatchPipeline, VideoPipeline
)

# Data structures
from .data import (
    Detection, Track, STrack, Pose, KeyPoint, ROI
)

# Utilities
from .utils import (
    Visualizer, Config, ResultExporter,
    PerformanceMonitor, Timer,
    VideoProcessor, VideoWriter, process_video
)

# Exceptions
from .exceptions import (
    VisionFrameworkError,
    DetectorInitializationError,
    DetectorInferenceError,
    TrackerInitializationError,
    TrackerUpdateError,
    ConfigurationError,
    ModelNotFoundError,
    ModelLoadError,
    DeviceError,
    DependencyError,
    DataFormatError,
    ProcessingError
)


# Simplified API functions
def create_detector(
    model_path: str = "yolov8n.pt",
    model_type: str = "yolo",
    device: str = "auto",
    conf_threshold: float = 0.25,
    **kwargs
) -> BaseDetector:
    """
    Create a detector with simplified parameters
    
    Args:
        model_path: Path to the model file or model name
        model_type: Type of model (yolo, detr, rfdetr)
        device: Device to use for inference
        conf_threshold: Confidence threshold for detections
        **kwargs: Additional configuration parameters
        
    Returns:
        Detector instance
    """
    config = {
        "model_path": model_path,
        "model_type": model_type,
        "device": device,
        "conf_threshold": conf_threshold,
        **kwargs
    }
    
    if model_type == "yolo":
        detector = YOLODetector(config)
    elif model_type == "detr":
        detector = DETRDetector(config)
    elif model_type == "rfdetr":
        detector = RFDETRDetector(config)
    else:
        detector = BaseDetector(config)
    
    detector.initialize()
    return detector


def create_pipeline(
    detector_config: Optional[Dict[str, Any]] = None,
    enable_tracking: bool = False,
    tracker_config: Optional[Dict[str, Any]] = None,
    enable_segmentation: bool = False,
    segmenter_config: Optional[Dict[str, Any]] = None,
    enable_pose_estimation: bool = False,
    pose_estimator_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> VisionPipeline:
    """
    Create a vision pipeline with simplified parameters
    
    Args:
        detector_config: Detector configuration
        enable_tracking: Whether to enable tracking
        tracker_config: Tracker configuration
        enable_segmentation: Whether to enable segmentation
        segmenter_config: Segmenter configuration
        enable_pose_estimation: Whether to enable pose estimation
        pose_estimator_config: Pose estimator configuration
        **kwargs: Additional configuration parameters
        
    Returns:
        VisionPipeline instance
    """
    config = {
        "detector_config": detector_config or {"model_path": "yolov8n.pt"},
        "enable_tracking": enable_tracking,
        "tracker_config": tracker_config or {},
        "enable_segmentation": enable_segmentation,
        "segmenter_config": segmenter_config or {},
        "enable_pose_estimation": enable_pose_estimation,
        "pose_estimator_config": pose_estimator_config or {},
        **kwargs
    }
    
    pipeline = VisionPipeline(config)
    pipeline.initialize()
    return pipeline


def process_image(
    image: np.ndarray,
    model_path: str = "yolov8n.pt",
    enable_tracking: bool = False,
    enable_segmentation: bool = False,
    enable_pose_estimation: bool = False,
    **kwargs
) -> Dict[str, Any]:
    """
    Process a single image with simplified parameters
    
    Args:
        image: Input image
        model_path: Path to the model file or model name
        enable_tracking: Whether to enable tracking
        enable_segmentation: Whether to enable segmentation
        enable_pose_estimation: Whether to enable pose estimation
        **kwargs: Additional configuration parameters
        
    Returns:
        Processing results
    """
    pipeline = create_pipeline(
        detector_config={"model_path": model_path, **kwargs},
        enable_tracking=enable_tracking,
        enable_segmentation=enable_segmentation,
        enable_pose_estimation=enable_pose_estimation
    )
    return pipeline.process(image)


def process_video(
    input_source: Union[str, int],
    output_path: Optional[str] = None,
    model_path: str = "yolov8n.pt",
    enable_tracking: bool = False,
    enable_segmentation: bool = False,
    enable_pose_estimation: bool = False,
    batch_size: int = 0,
    use_pyav: bool = False,
    **kwargs
) -> bool:
    """
    Run video processing with minimal configuration
    
    This static method provides a convenient way to process videos, camera streams, or network streams
    with just one line of code.
    
    Args:
        input_source: Path to video file, video stream URL, or camera index
        output_path: Optional path to save processed video
        model_path: Path to the detection model (default: "yolov8n.pt")
        enable_tracking: Whether to enable tracking (default: False)
        enable_segmentation: Whether to enable segmentation
        enable_pose_estimation: Whether to enable pose estimation
        batch_size: Batch size for processing (0 for non-batch processing, default: 0)
        use_pyav: Whether to use pyav for video processing (default: False, use OpenCV)
                  Note: PyAV supports video files and streams, but not cameras
        **kwargs: Additional arguments passed to process_video
        
    Returns:
        bool: True if processing completed successfully, False otherwise
    """
    pipeline = create_pipeline(
        detector_config={"model_path": model_path, **kwargs},
        enable_tracking=enable_tracking,
        enable_segmentation=enable_segmentation,
        enable_pose_estimation=enable_pose_estimation
    )
    
    if batch_size > 0:
        return pipeline.process_video_batch(
            input_source, output_path, batch_size=batch_size, use_pyav=use_pyav
        )
    else:
        return pipeline.process_video(
            input_source, output_path, use_pyav=use_pyav
        )


def create_visualizer(
    config: Optional[Dict[str, Any]] = None
) -> Visualizer:
    """
    Create a visualizer with simplified parameters
    
    Args:
        config: Visualizer configuration
        
    Returns:
        Visualizer instance
    """
    return Visualizer(config)


# Export all symbols to make them available directly from visionframework.api
__all__ = [
    # Core
    "BaseDetector",
    "BaseTracker",
    "VisionPipeline",
    "ROIDetector",
    "Counter",
    "SAMSegmenter",
    
    # Processors
    "PoseEstimator",
    "CLIPExtractor",
    "ReIDExtractor",
    
    # Implementations
    "YOLODetector",
    "DETRDetector",
    "RFDETRDetector",
    "IOUTracker",
    "ByteTracker",
    "ReIDTracker",
    
    # Data structures
    "Detection",
    "Track",
    "STrack",
    "Pose",
    "KeyPoint",
    "ROI",
    
    # Utilities
    "Visualizer",
    "Config",
    "ResultExporter",
    "PerformanceMonitor",
    "Timer",
    "VideoProcessor",
    "VideoWriter",
    "process_video",
    
    # Plugin system
    "plugin_registry",
    "model_registry",
    "register_detector",
    "register_tracker",
    "register_segmenter",
    "register_model",
    "register_processor",
    "register_visualizer",
    "register_evaluator",
    "register_custom_component",
    
    # Exceptions
    "VisionFrameworkError",
    "DetectorInitializationError",
    "DetectorInferenceError",
    "TrackerInitializationError",
    "TrackerUpdateError",
    "ConfigurationError",
    "ModelNotFoundError",
    "ModelLoadError",
    "DeviceError",
    "DependencyError",
    "DataFormatError",
    "ProcessingError",
    
    # Pipeline classes
    "BasePipeline",
    "VisionPipeline",
    "BatchPipeline",
    "VideoPipeline",
    
    # Simplified API functions
    "create_detector",
    "create_pipeline",
    "process_image",
    "process_video",
    "create_visualizer",
    
    # Version
    # "__version__",  # Defined in visionframework/__init__.py
]
