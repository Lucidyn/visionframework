"""
Detector implementations with lazy loading to avoid heavy imports
"""

from .base_detector import BaseDetector

def __getattr__(name):
    """Lazy load detector implementations to avoid importing heavy libraries at module load time"""
    if name == "YOLODetector":
        from .yolo_detector import YOLODetector
        return YOLODetector
    elif name == "DETRDetector":
        from .detr_detector import DETRDetector
        return DETRDetector
    elif name == "RFDETRDetector":
        from .rfdetr_detector import RFDETRDetector
        return RFDETRDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseDetector",
    "YOLODetector",
    "DETRDetector",
    "RFDETRDetector",
]

