"""
Core infrastructure: registry, builder, and configuration system.
"""

from .registry import Registry, BACKBONES, NECKS, HEADS, ALGORITHMS, PIPELINES
from .config import load_config, merge_configs, require_detector_weights
from .builder import build_model, build_algorithm, build_pipeline

__all__ = [
    "Registry",
    "BACKBONES", "NECKS", "HEADS", "ALGORITHMS", "PIPELINES",
    "load_config", "merge_configs", "require_detector_weights",
    "build_model", "build_algorithm", "build_pipeline",
]
