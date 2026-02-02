"""
Model optimization utilities

This module provides tools for model quantization, pruning, and optimization.
"""

from .quantization import (
    quantize_model,
    get_quantized_model,
    compare_model_performance,
    QuantizationConfig
)

from .pruning import (
    prune_model,
    get_pruned_model,
    apply_pruning,
    PruningConfig
)

from .distillation import (
    distill_model,
    get_distilled_model,
    DistillationConfig
)

__all__ = [
    "quantize_model",
    "get_quantized_model",
    "compare_model_performance",
    "QuantizationConfig",
    "prune_model",
    "get_pruned_model",
    "apply_pruning",
    "PruningConfig",
    "distill_model",
    "get_distilled_model",
    "DistillationConfig"
]
