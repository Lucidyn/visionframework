"""
Model format definitions and utilities

This module defines supported model formats and provides utility functions for format handling.
"""

from enum import Enum
from typing import List, Dict, Any


class ModelFormat(Enum):
    """
    Supported model formats
    """
    # PyTorch formats
    PYTORCH = "pytorch"
    TORCHSCRIPT = "torchscript"
    
    # ONNX format
    ONNX = "onnx"
    
    # TensorRT formats
    TENSORRT = "tensorrt"
    TENSORRT_ENGINE = "tensorrt_engine"
    
    # TensorFlow formats
    TENSORFLOW = "tensorflow"
    TFLITE = "tflite"
    SAVED_MODEL = "saved_model"
    
    # OpenVINO format
    OPENVINO = "openvino"
    
    # CoreML format
    COREML = "coreml"
    
    # ONNX Runtime format
    ONNX_RUNTIME = "onnx_runtime"


# Format compatibility matrix
FORMAT_COMPATIBILITY: Dict[ModelFormat, List[ModelFormat]] = {
    ModelFormat.PYTORCH: [
        ModelFormat.TORCHSCRIPT,
        ModelFormat.ONNX,
        ModelFormat.TFLITE,
        ModelFormat.OPENVINO,
        ModelFormat.COREML
    ],
    ModelFormat.TORCHSCRIPT: [
        ModelFormat.ONNX
    ],
    ModelFormat.ONNX: [
        ModelFormat.TENSORRT,
        ModelFormat.TENSORRT_ENGINE,
        ModelFormat.OPENVINO,
        ModelFormat.ONNX_RUNTIME
    ],
    ModelFormat.TENSORFLOW: [
        ModelFormat.TFLITE,
        ModelFormat.SAVED_MODEL,
        ModelFormat.ONNX
    ],
    ModelFormat.SAVED_MODEL: [
        ModelFormat.ONNX,
        ModelFormat.TFLITE
    ]
}

# Format file extensions
FORMAT_EXTENSIONS: Dict[ModelFormat, str] = {
    ModelFormat.PYTORCH: ".pt",
    ModelFormat.TORCHSCRIPT: ".ts",
    ModelFormat.ONNX: ".onnx",
    ModelFormat.TENSORRT: ".plan",
    ModelFormat.TENSORRT_ENGINE: ".engine",
    ModelFormat.TENSORFLOW: ".pb",
    ModelFormat.TFLITE: ".tflite",
    ModelFormat.SAVED_MODEL: ".savedmodel",
    ModelFormat.OPENVINO: ".xml",
    ModelFormat.COREML: ".mlmodel",
    ModelFormat.ONNX_RUNTIME: ".onnx"
}

# Required dependencies for each format
FORMAT_DEPENDENCIES: Dict[ModelFormat, List[str]] = {
    ModelFormat.PYTORCH: ["torch"],
    ModelFormat.TORCHSCRIPT: ["torch"],
    ModelFormat.ONNX: ["torch", "onnx"],
    ModelFormat.TENSORRT: ["tensorrt", "onnx"],
    ModelFormat.TENSORRT_ENGINE: ["tensorrt", "onnx"],
    ModelFormat.TENSORFLOW: ["tensorflow"],
    ModelFormat.TFLITE: ["tensorflow", "tensorflow-lite"],
    ModelFormat.SAVED_MODEL: ["tensorflow"],
    ModelFormat.OPENVINO: ["openvino-dev"],
    ModelFormat.COREML: ["coremltools"],
    ModelFormat.ONNX_RUNTIME: ["onnxruntime"]
}


def get_supported_formats() -> List[ModelFormat]:
    """
    Get list of supported model formats
    
    Returns:
        List of supported ModelFormat enum values
    """
    return list(ModelFormat)


def is_format_supported(format: ModelFormat) -> bool:
    """
    Check if a model format is supported
    
    Args:
        format: Model format to check
    
    Returns:
        True if format is supported, False otherwise
    """
    try:
        # Check if format is in the enum
        ModelFormat(format)
        return True
    except ValueError:
        return False


def get_compatible_formats(source_format: ModelFormat) -> List[ModelFormat]:
    """
    Get list of formats compatible with the source format
    
    Args:
        source_format: Source model format
    
    Returns:
        List of compatible target formats
    """
    return FORMAT_COMPATIBILITY.get(source_format, [])


def get_format_extension(format: ModelFormat) -> str:
    """
    Get file extension for a model format
    
    Args:
        format: Model format
    
    Returns:
        File extension string
    """
    return FORMAT_EXTENSIONS.get(format, ".model")


def get_format_dependencies(format: ModelFormat) -> List[str]:
    """
    Get required dependencies for a model format
    
    Args:
        format: Model format
    
    Returns:
        List of required dependency package names
    """
    return FORMAT_DEPENDENCIES.get(format, [])


def get_format_from_extension(extension: str) -> ModelFormat:
    """
    Get model format from file extension
    
    Args:
        extension: File extension string (with or without dot)
    
    Returns:
        Corresponding ModelFormat enum value
    
    Raises:
        ValueError: If extension is not recognized
    """
    # Remove dot if present
    if extension.startswith("."):
        extension = extension[1:]
    
    # Reverse lookup in FORMAT_EXTENSIONS
    for format, ext in FORMAT_EXTENSIONS.items():
        if ext.lstrip(".") == extension:
            return format
    
    raise ValueError(f"Unrecognized file extension: .{extension}")
