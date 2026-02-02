"""
Deployment platform definitions and utilities

This module defines supported deployment platforms and provides utility functions for platform handling.
"""

from enum import Enum
from typing import List, Dict, Any


class DeploymentPlatform(Enum):
    """
    Supported deployment platforms
    """
    # Local platforms
    LOCAL = "local"
    DOCKER = "docker"
    
    # Cloud platforms
    AWS_SAGEMAKER = "aws_sagemaker"
    AZURE_ML = "azure_ml"
    GCP_AI_PLATFORM = "gcp_ai_platform"
    
    # Edge platforms
    NVIDIA_JETSON = "nvidia_jetson"
    RASPBERRY_PI = "raspberry_pi"
    EDGE_DEVICE = "edge_device"
    
    # Web platforms
    WEB = "web"
    WEB_ASSEMBLY = "web_assembly"
    
    # Mobile platforms
    ANDROID = "android"
    IOS = "ios"
    
    # Embedded platforms
    EMBEDDED = "embedded"
    MICROCONTROLLER = "microcontroller"


# Platform compatibility matrix
PLATFORM_COMPATIBILITY: Dict[DeploymentPlatform, List[str]] = {
    DeploymentPlatform.LOCAL: [
        "pytorch",
        "onnx",
        "tensorrt",
        "tflite",
        "openvino"
    ],
    DeploymentPlatform.DOCKER: [
        "pytorch",
        "onnx",
        "tensorrt",
        "tflite",
        "openvino"
    ],
    DeploymentPlatform.AWS_SAGEMAKER: [
        "pytorch",
        "onnx",
        "tensorflow"
    ],
    DeploymentPlatform.AZURE_ML: [
        "pytorch",
        "onnx",
        "tensorflow"
    ],
    DeploymentPlatform.GCP_AI_PLATFORM: [
        "pytorch",
        "onnx",
        "tensorflow"
    ],
    DeploymentPlatform.NVIDIA_JETSON: [
        "tensorrt",
        "onnx",
        "pytorch"
    ],
    DeploymentPlatform.RASPBERRY_PI: [
        "tflite",
        "onnx",
        "openvino"
    ],
    DeploymentPlatform.EDGE_DEVICE: [
        "tflite",
        "onnx",
        "openvino"
    ],
    DeploymentPlatform.WEB: [
        "onnx",
        "tensorflow.js"
    ],
    DeploymentPlatform.WEB_ASSEMBLY: [
        "onnx",
        "tensorflow.js"
    ],
    DeploymentPlatform.ANDROID: [
        "tflite",
        "onnx"
    ],
    DeploymentPlatform.IOS: [
        "coreml",
        "tflite",
        "onnx"
    ],
    DeploymentPlatform.EMBEDDED: [
        "tflite",
        "onnx"
    ],
    DeploymentPlatform.MICROCONTROLLER: [
        "tflite_micro"
    ]
}

# Platform requirements
PLATFORM_REQUIREMENTS: Dict[DeploymentPlatform, Dict[str, Any]] = {
    DeploymentPlatform.LOCAL: {
        "dependencies": [],
        "hardware": {}
    },
    DeploymentPlatform.DOCKER: {
        "dependencies": ["docker"],
        "hardware": {}
    },
    DeploymentPlatform.AWS_SAGEMAKER: {
        "dependencies": ["boto3", "sagemaker"],
        "hardware": {}
    },
    DeploymentPlatform.AZURE_ML: {
        "dependencies": ["azureml-core"],
        "hardware": {}
    },
    DeploymentPlatform.GCP_AI_PLATFORM: {
        "dependencies": ["google-cloud-aiplatform"],
        "hardware": {}
    },
    DeploymentPlatform.NVIDIA_JETSON: {
        "dependencies": ["tensorrt", "torch2trt"],
        "hardware": {
            "gpu": "nvidia",
            "architecture": "arm64"
        }
    },
    DeploymentPlatform.RASPBERRY_PI: {
        "dependencies": ["tflite-runtime"],
        "hardware": {
            "architecture": "armv7"
        }
    },
    DeploymentPlatform.EDGE_DEVICE: {
        "dependencies": ["tflite-runtime", "openvino-dev"],
        "hardware": {}
    },
    DeploymentPlatform.WEB: {
        "dependencies": ["onnxruntime-web", "tensorflowjs"],
        "hardware": {}
    },
    DeploymentPlatform.WEB_ASSEMBLY: {
        "dependencies": ["emscripten"],
        "hardware": {}
    },
    DeploymentPlatform.ANDROID: {
        "dependencies": ["tensorflow-lite", "onnxruntime-android"],
        "hardware": {
            "architecture": "arm64-v8a"
        }
    },
    DeploymentPlatform.IOS: {
        "dependencies": ["coremltools", "tensorflow-lite", "onnxruntime-ios"],
        "hardware": {
            "architecture": "arm64"
        }
    },
    DeploymentPlatform.EMBEDDED: {
        "dependencies": ["tflite-micro"],
        "hardware": {
            "constraints": "low_power"
        }
    },
    DeploymentPlatform.MICROCONTROLLER: {
        "dependencies": ["tflite-micro"],
        "hardware": {
            "constraints": "ultra_low_power"
        }
    }
}


def get_supported_platforms() -> List[DeploymentPlatform]:
    """
    Get list of supported deployment platforms
    
    Returns:
        List of supported DeploymentPlatform enum values
    """
    return list(DeploymentPlatform)


def is_platform_supported(platform: DeploymentPlatform) -> bool:
    """
    Check if a deployment platform is supported
    
    Args:
        platform: Deployment platform to check
    
    Returns:
        True if platform is supported, False otherwise
    """
    try:
        # Check if platform is in the enum
        DeploymentPlatform(platform)
        return True
    except ValueError:
        return False


def get_platform_compatibility(platform: DeploymentPlatform) -> List[str]:
    """
    Get list of model formats compatible with the platform
    
    Args:
        platform: Deployment platform
    
    Returns:
        List of compatible model formats
    """
    return PLATFORM_COMPATIBILITY.get(platform, [])


def get_platform_requirements(platform: DeploymentPlatform) -> Dict[str, Any]:
    """
    Get requirements for a deployment platform
    
    Args:
        platform: Deployment platform
    
    Returns:
        Dictionary with platform requirements
    """
    return PLATFORM_REQUIREMENTS.get(platform, {"dependencies": [], "hardware": {}})


def get_platform_from_string(platform_str: str) -> DeploymentPlatform:
    """
    Get deployment platform from string
    
    Args:
        platform_str: Platform string
    
    Returns:
        Corresponding DeploymentPlatform enum value
    
    Raises:
        ValueError: If platform string is not recognized
    """
    try:
        return DeploymentPlatform(platform_str)
    except ValueError:
        raise ValueError(f"Unrecognized platform: {platform_str}")
