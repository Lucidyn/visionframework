"""
Multimodal fusion utilities

This module provides utilities for multimodal fusion, including techniques for
combining information from different modalities like vision and language.
"""

from .fusion import (
    MultimodalFusion,
    FusionConfig,
    fuse_features,
    get_fusion_model
)

__all__ = [
    "MultimodalFusion",
    "FusionConfig",
    "fuse_features",
    "get_fusion_model"
]
