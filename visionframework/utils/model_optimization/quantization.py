"""
Model quantization utilities

This module provides tools for model quantization to reduce model size and improve inference speed.
"""

import torch
import time
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.ao.quantization import get_default_qconfig_mapping

try:
    # Newer PyTorch versions provide backend-specific default backend configs.
    from torch.ao.quantization.backend_config import get_default_backend_config  # type: ignore
except Exception:  # pragma: no cover - fallback for older Torch versions
    get_default_backend_config = None  # type: ignore


@dataclass
class QuantizationConfig:
    """
    Configuration for model quantization
    
    Attributes:
        quantization_type: Type of quantization ('dynamic', 'static', 'aware')
        backend: Quantization backend ('qnnpack', 'fbgemm', 'onednn')
        dtype: Quantization dtype (typically torch.qint8 for dynamic quantization)
        qconfig_mapping: Custom qconfig mapping
        calibration_data: Data for static quantization calibration
        verbose: Whether to print verbose information
    """
    quantization_type: str = "dynamic"
    backend: str = "fbgemm"
    # Use qint8 by default to be compatible with dynamic quantized linear layers
    dtype: torch.dtype = torch.qint8
    qconfig_mapping: Optional[Any] = None
    calibration_data: Optional[Any] = None
    verbose: bool = False


def quantize_model(model: torch.nn.Module, config: QuantizationConfig) -> torch.nn.Module:
    """
    Quantize a PyTorch model
    
    Args:
        model: PyTorch model to quantize
        config: Quantization configuration
    
    Returns:
        Quantized PyTorch model
    """
    if config.verbose:
        print(f"Quantizing model with {config.quantization_type} quantization")
    
    # Set quantization backend
    torch.backends.quantized.engine = config.backend
    
    if config.quantization_type == "dynamic":
        return _quantize_dynamic(model, config)
    elif config.quantization_type == "static":
        return _quantize_static(model, config)
    elif config.quantization_type == "aware":
        return _quantize_aware(model, config)
    else:
        raise ValueError(f"Unsupported quantization type: {config.quantization_type}")


def _quantize_dynamic(model: torch.nn.Module, config: QuantizationConfig) -> torch.nn.Module:
    """
    Apply dynamic quantization
    
    Args:
        model: PyTorch model to quantize
        config: Quantization configuration
    
    Returns:
        Dynamically quantized PyTorch model
    """
    if config.verbose:
        print("Applying dynamic quantization")
    
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=config.dtype
    )
    
    return quantized_model


def _quantize_static(model: torch.nn.Module, config: QuantizationConfig) -> torch.nn.Module:
    """
    Apply static quantization
    
    Args:
        model: PyTorch model to quantize
        config: Quantization configuration
    
    Returns:
        Statically quantized PyTorch model
    """
    if config.verbose:
        print("Applying static quantization")
    
    # Get default qconfig mapping if not provided
    if config.qconfig_mapping is None:
        qconfig_mapping = get_default_qconfig_mapping(config.backend)
    else:
        qconfig_mapping = config.qconfig_mapping
    
    # Get default backend config if available (PyTorch >= 1.13 / 2.x).
    if get_default_backend_config is None:
        raise RuntimeError(
            "Static quantization is not supported in this PyTorch version: "
            "missing get_default_backend_config in torch.ao.quantization.backend_config."
        )
    backend_config = get_default_backend_config(config.backend)
    
    # Prepare model for quantization
    example_inputs = next(iter(config.calibration_data)) if config.calibration_data else torch.randn(1, 3, 224, 224)
    prepared_model = prepare_fx(model, qconfig_mapping, example_inputs)
    
    # Calibrate model
    if config.calibration_data:
        if config.verbose:
            print("Calibrating model...")
        for data in config.calibration_data:
            prepared_model(data)
    
    # Convert to quantized model
    quantized_model = convert_fx(prepared_model, backend_config)
    
    return quantized_model


def _quantize_aware(model: torch.nn.Module, config: QuantizationConfig) -> torch.nn.Module:
    """
    Apply quantization-aware training
    
    Args:
        model: PyTorch model to quantize
        config: Quantization configuration
    
    Returns:
        Quantization-aware trained PyTorch model
    """
    if config.verbose:
        print("Preparing model for quantization-aware training")
    
    # Check if model has quant stubs
    if not hasattr(model, 'quant'):
        # Add quant and dequant stubs
        model.quant = QuantStub()
        model.dequant = DeQuantStub()
        
        # Modify forward method
        original_forward = model.forward
        
        def quantized_forward(self, x):
            x = self.quant(x)
            x = original_forward(x)
            x = self.dequant(x)
            return x
        
        model.forward = quantized_forward.__get__(model)
    
    # Get qconfig
    if config.backend == "fbgemm":
        qconfig = torch.quantization.get_default_qconfig(config.backend)
    else:
        qconfig = torch.quantization.default_qconfig
    
    # Apply quantization config
    model.qconfig = qconfig
    
    # Prepare model
    model = torch.quantization.prepare_qat(model)
    
    return model


def get_quantized_model(model_path: str, config: QuantizationConfig) -> torch.nn.Module:
    """
    Load and quantize a model from file
    
    Args:
        model_path: Path to model file
        config: Quantization configuration
    
    Returns:
        Quantized PyTorch model
    """
    # Load model
    model = torch.load(model_path)
    
    # Quantize model
    quantized_model = quantize_model(model, config)
    
    return quantized_model


def compare_model_performance(
    original_model: torch.nn.Module, 
    optimized_model: torch.nn.Module, 
    test_data: Any
) -> Dict[str, Any]:
    """
    Compare performance of original and optimized models
    
    Args:
        original_model: Original PyTorch model
        optimized_model: Optimized PyTorch model
        test_data: Test data for evaluation
    
    Returns:
        Dictionary with performance metrics
    """
    # Measure model size
    import os
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(original_model, f.name)
        original_size = os.path.getsize(f.name)
    os.unlink(f.name)
    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(optimized_model, f.name)
        optimized_size = os.path.getsize(f.name)
    os.unlink(f.name)
    
    # Measure inference time
    original_times = []
    optimized_times = []
    
    # Move models to evaluation mode
    original_model.eval()
    optimized_model.eval()
    
    # Measure inference time
    with torch.no_grad():
        for data in test_data:
            # Original model
            start_time = time.time()
            _ = original_model(data)
            original_times.append(time.time() - start_time)
            
            # Optimized model
            start_time = time.time()
            _ = optimized_model(data)
            optimized_times.append(time.time() - start_time)
    
    # Calculate metrics
    original_time = sum(original_times) / len(original_times) if original_times else 0.0
    optimized_time = sum(optimized_times) / len(optimized_times) if optimized_times else 0.0
    if optimized_time > 0:
        speedup = original_time / optimized_time
    else:
        speedup = 1.0
    
    metrics = {
        "original_size": original_size,
        "optimized_size": optimized_size,
        "size_reduction": (original_size - optimized_size) / original_size * 100,
        "original_inference_time": original_time,
        "optimized_inference_time": optimized_time,
        "speedup": speedup
    }
    
    return metrics
