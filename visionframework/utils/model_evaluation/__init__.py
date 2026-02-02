"""
Model evaluation utilities

This module provides utilities for model evaluation, including metrics calculation,
benchmarking, and performance analysis.
"""

from .evaluator import (
    ModelEvaluator,
    EvaluationConfig,
    evaluate_model
)

__all__ = [
    "ModelEvaluator",
    "EvaluationConfig",
    "evaluate_model"
]
