"""
Tracker implementations with lazy loading for heavy dependencies
"""

from .base_tracker import BaseTracker


def __getattr__(name):
    """Lazy load tracker implementations to avoid importing heavy libraries at module load time"""
    if name == "IOUTracker":
        from .iou_tracker import IOUTracker
        return IOUTracker
    elif name == "ByteTracker":
        from .byte_tracker import ByteTracker
        return ByteTracker
    elif name == "ReIDTracker":
        from .reid_tracker import ReIDTracker
        return ReIDTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseTracker",
    "IOUTracker",
    "ByteTracker",
    "ReIDTracker",
]

