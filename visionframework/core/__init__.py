"""
Core module for Vision Framework
"""

# 只导入最基本的模块，避免循环导入
from .base import BaseModule

# 导入pipeline类
from .pipelines import (
    BasePipeline,
    VisionPipeline,
    BatchPipeline,
    VideoPipeline
)

# 延迟导入其他模块，避免循环导入
def __getattr__(name):
    """Lazy load modules to avoid circular dependencies"""
    # Core components
    if name == "ROIDetector":
        from .roi_detector import ROIDetector
        return ROIDetector
    elif name == "Counter":
        from .counter import Counter
        return Counter
    
    # Segmenters
    elif name == "SAMSegmenter":
        from .components.segmenters import SAMSegmenter
        return SAMSegmenter
    elif name == "BaseSegmenter":
        from .components.segmenters import BaseSegmenter
        return BaseSegmenter
    
    # Detectors
    elif name == "BaseDetector":
        from .components.detectors import BaseDetector
        return BaseDetector
    elif name == "YOLODetector":
        from .components.detectors.yolo_detector import YOLODetector
        return YOLODetector
    elif name == "DETRDetector":
        from .components.detectors.detr_detector import DETRDetector
        return DETRDetector
    elif name == "RFDETRDetector":
        from .components.detectors.rfdetr_detector import RFDETRDetector
        return RFDETRDetector
    
    # Trackers
    elif name == "BaseTracker":
        from .components.trackers import BaseTracker
        return BaseTracker
    elif name == "IOUTracker":
        from .components.trackers.iou_tracker import IOUTracker
        return IOUTracker
    elif name == "ByteTracker":
        from .components.trackers.byte_tracker import ByteTracker
        return ByteTracker
    elif name == "ReIDTracker":
        from .components.trackers.reid_tracker import ReIDTracker
        return ReIDTracker
    
    # Processors
    elif name == "FeatureExtractor":
        from .components.processors import FeatureExtractor
        return FeatureExtractor
    elif name == "CLIPExtractor":
        from .components.processors.clip_extractor import CLIPExtractor
        return CLIPExtractor
    elif name == "ReIDExtractor":
        from .components.processors.reid_extractor import ReIDExtractor
        return ReIDExtractor
    
    # Plugin system
    elif name == "plugin_registry":
        from .plugin_system import plugin_registry
        return plugin_registry
    elif name == "model_registry":
        from .plugin_system import model_registry
        return model_registry
    elif name == "register_detector":
        from .plugin_system import register_detector
        return register_detector
    elif name == "register_tracker":
        from .plugin_system import register_tracker
        return register_tracker
    elif name == "register_segmenter":
        from .plugin_system import register_segmenter
        return register_segmenter
    elif name == "register_model":
        from .plugin_system import register_model
        return register_model
    elif name == "register_processor":
        from .plugin_system import register_processor
        return register_processor
    elif name == "register_visualizer":
        from .plugin_system import register_visualizer
        return register_visualizer
    elif name == "register_evaluator":
        from .plugin_system import register_evaluator
        return register_evaluator
    elif name == "register_custom_component":
        from .plugin_system import register_custom_component
        return register_custom_component
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Export all symbols
__all__ = [
    # Base classes
    "BaseModule",
    "BasePipeline",
    
    # Pipeline classes
    "VisionPipeline",
    "BatchPipeline",
    "VideoPipeline",
    
    # Core components
    "ROIDetector",
    "Counter",
    "SAMSegmenter",
    
    # Base component classes
    "BaseDetector",
    "BaseTracker",
    "BaseSegmenter",
    "FeatureExtractor",
    
    # Implementations
    "YOLODetector",
    "DETRDetector",
    "RFDETRDetector",
    "IOUTracker",
    "ByteTracker",
    "ReIDTracker",
    "CLIPExtractor",
    "ReIDExtractor",
    
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
    "register_custom_component"
]
