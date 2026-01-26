"""
Data processing utilities
"""

from .export import ResultExporter
from .image_utils import ImageUtils
from .trajectory_analyzer import TrajectoryAnalyzer

__all__ = [
    "AutoLabeler",
    "ResultExporter",
    "ImageUtils",
    "TrajectoryAnalyzer",
]


def __getattr__(name):
    """Lazy load modules to avoid circular imports"""
    if name == "AutoLabeler":
        from .auto_labeler import AutoLabeler
        return AutoLabeler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
