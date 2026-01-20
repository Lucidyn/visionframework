"""
Feature Extractors with lazy loading for heavy dependencies

Provides feature extraction modules for various tasks including CLIP,
pose estimation, and ReID.
"""

from .feature_extractor import FeatureExtractor


def __getattr__(name):
    """Lazy load heavy processor implementations to avoid importing heavy libraries at module load time"""
    if name == "CLIPExtractor":
        from .clip_extractor import CLIPExtractor
        return CLIPExtractor
    elif name == "ReIDExtractor":
        from .reid_extractor import ReIDExtractor
        return ReIDExtractor
    elif name == "PoseEstimator":
        from .pose_estimator import PoseEstimator
        return PoseEstimator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FeatureExtractor",
    "CLIPExtractor",
    "ReIDExtractor",
    "PoseEstimator",
]
