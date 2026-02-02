"""
Model pruning utilities

This module provides tools for model pruning to reduce model size and improve inference speed.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from torch.nn.utils import prune


@dataclass
class PruningConfig:
    """
    Configuration for model pruning
    
    Attributes:
        pruning_type: Type of pruning ('l1_unstructured', 'l2_unstructured', 'random_unstructured', 'ln_structured')
        amount: Pruning amount (0.0 to 1.0)
        target_modules: List of module types to prune
        global_pruning: Whether to apply global pruning
        verbose: Whether to print verbose information
    """
    pruning_type: str = "l1_unstructured"
    amount: float = 0.2
    target_modules: List[type] = None
    global_pruning: bool = False
    verbose: bool = False

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [nn.Linear, nn.Conv2d]


def prune_model(model: nn.Module, config: PruningConfig) -> nn.Module:
    """
    Prune a PyTorch model
    
    Args:
        model: PyTorch model to prune
        config: Pruning configuration
    
    Returns:
        Pruned PyTorch model
    """
    if config.verbose:
        print(f"Pruning model with {config.pruning_type} pruning")
    
    if config.global_pruning:
        return _apply_global_pruning(model, config)
    else:
        return _apply_local_pruning(model, config)


def _apply_local_pruning(model: nn.Module, config: PruningConfig) -> nn.Module:
    """
    Apply local pruning to each module individually
    
    Args:
        model: PyTorch model to prune
        config: Pruning configuration
    
    Returns:
        Pruned PyTorch model
    """
    if config.verbose:
        print("Applying local pruning")
    
    # Iterate through all modules
    for name, module in model.named_modules():
        if type(module) in config.target_modules:
            if config.verbose:
                print(f"Pruning module: {name}")
            
            # Apply pruning based on type
            if config.pruning_type == "l1_unstructured":
                prune.l1_unstructured(module, name="weight", amount=config.amount)
            elif config.pruning_type == "l2_unstructured":
                prune.l2_unstructured(module, name="weight", amount=config.amount)
            elif config.pruning_type == "random_unstructured":
                prune.random_unstructured(module, name="weight", amount=config.amount)
            elif config.pruning_type == "ln_structured":
                prune.ln_structured(module, name="weight", amount=config.amount, n=2, dim=0)
            else:
                raise ValueError(f"Unsupported pruning type: {config.pruning_type}")
    
    # Remove pruning reparameterization
    for name, module in model.named_modules():
        if type(module) in config.target_modules:
            prune.remove(module, "weight")
    
    return model


def _apply_global_pruning(model: nn.Module, config: PruningConfig) -> nn.Module:
    """
    Apply global pruning across all modules
    
    Args:
        model: PyTorch model to prune
        config: Pruning configuration
    
    Returns:
        Pruned PyTorch model
    """
    if config.verbose:
        print("Applying global pruning")
    
    # Collect all parameters to prune
    parameters_to_prune = []
    for name, module in model.named_modules():
        if type(module) in config.target_modules:
            parameters_to_prune.append((module, "weight"))
    
    # Apply global pruning
    if config.pruning_type == "l1_unstructured":
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=config.amount,
        )
    elif config.pruning_type == "l2_unstructured":
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L2Unstructured,
            amount=config.amount,
        )
    elif config.pruning_type == "random_unstructured":
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=config.amount,
        )
    else:
        raise ValueError(f"Unsupported global pruning type: {config.pruning_type}")
    
    # Remove pruning reparameterization
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    return model


def get_pruned_model(model_path: str, config: PruningConfig) -> nn.Module:
    """
    Load and prune a model from file
    
    Args:
        model_path: Path to model file
        config: Pruning configuration
    
    Returns:
        Pruned PyTorch model
    """
    # Load model
    model = torch.load(model_path)
    
    # Prune model
    pruned_model = prune_model(model, config)
    
    return pruned_model


def apply_pruning(model: nn.Module, config: PruningConfig) -> nn.Module:
    """
    Apply pruning to a model and return the pruned model
    
    Args:
        model: PyTorch model to prune
        config: Pruning configuration
    
    Returns:
        Pruned PyTorch model
    """
    return prune_model(model, config)
