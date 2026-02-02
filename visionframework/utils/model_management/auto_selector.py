"""
Auto model selection functionality

This module provides functionality for automatically selecting the most appropriate model
based on user requirements and hardware constraints.
"""

import os
import platform
import subprocess
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum


class ModelType(Enum):
    """Model type enum"""
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    POSE = "pose"
    REID = "reid"
    CLIP = "clip"
    FACE = "face"
    DETR = "detr"
    RFDETR = "rfdetr"
    SAM = "sam"


class HardwareTier(Enum):
    """Hardware tier enum"""
    HIGH_END = "high_end"
    MID_RANGE = "mid_range"
    LOW_END = "low_end"
    EDGE = "edge"
    MOBILE = "mobile"


@dataclass
class ModelRequirement:
    """
    Model requirement class
    
    Attributes:
        model_type: Type of model needed
        accuracy: Required accuracy level (0-100)
        speed: Required speed level (0-100)
        memory: Required memory in MB
        task: Specific task description
        constraints: Additional constraints
    """
    model_type: ModelType
    accuracy: int = 70
    speed: int = 50
    memory: int = 1024
    task: str = "general"
    constraints: Optional[Dict[str, Any]] = None


@dataclass
class HardwareInfo:
    """
    Hardware information class
    
    Attributes:
        platform: Operating system platform
        cpu_cores: Number of CPU cores
        cpu_ram: CPU RAM in MB
        gpu_available: Whether GPU is available
        gpu_memory: GPU memory in MB
        gpu_name: GPU name
        hardware_tier: Hardware tier
    """
    platform: str
    cpu_cores: int
    cpu_ram: int
    gpu_available: bool
    gpu_memory: int
    gpu_name: str
    hardware_tier: HardwareTier


class ModelSelector:
    """
    Model selector class for automatically selecting the most appropriate model
    based on user requirements and hardware constraints.
    """
    
    def __init__(self):
        """
        Initialize model selector
        """
        self._model_database = self._build_model_database()
        self._hardware_info = self._detect_hardware()
    
    def _build_model_database(self) -> Dict[str, Dict[str, Any]]:
        """
        Build model database with model characteristics
        
        Returns:
            Dictionary of model information
        """
        return {
            # Detection models
            "yolov8n": {
                "type": ModelType.DETECTION,
                "accuracy": 65,
                "speed": 95,
                "memory": 120,
                "size": 3.1,
                "hardware_tier": HardwareTier.MOBILE
            },
            "yolov8s": {
                "type": ModelType.DETECTION,
                "accuracy": 75,
                "speed": 85,
                "memory": 250,
                "size": 11.2,
                "hardware_tier": HardwareTier.LOW_END
            },
            "yolov8m": {
                "type": ModelType.DETECTION,
                "accuracy": 83,
                "speed": 65,
                "memory": 500,
                "size": 25.9,
                "hardware_tier": HardwareTier.MID_RANGE
            },
            "yolov8l": {
                "type": ModelType.DETECTION,
                "accuracy": 87,
                "speed": 45,
                "memory": 800,
                "size": 46.5,
                "hardware_tier": HardwareTier.HIGH_END
            },
            "yolov8x": {
                "type": ModelType.DETECTION,
                "accuracy": 90,
                "speed": 30,
                "memory": 1200,
                "size": 86.7,
                "hardware_tier": HardwareTier.HIGH_END
            },
            "yolov26n": {
                "type": ModelType.DETECTION,
                "accuracy": 70,
                "speed": 90,
                "memory": 150,
                "size": 4.5,
                "hardware_tier": HardwareTier.MOBILE
            },
            "yolov26s": {
                "type": ModelType.DETECTION,
                "accuracy": 80,
                "speed": 80,
                "memory": 300,
                "size": 16.8,
                "hardware_tier": HardwareTier.LOW_END
            },
            "yolov26m": {
                "type": ModelType.DETECTION,
                "accuracy": 86,
                "speed": 60,
                "memory": 600,
                "size": 38.4,
                "hardware_tier": HardwareTier.MID_RANGE
            },
            "yolov26l": {
                "type": ModelType.DETECTION,
                "accuracy": 89,
                "speed": 40,
                "memory": 900,
                "size": 67.2,
                "hardware_tier": HardwareTier.HIGH_END
            },
            "yolov26x": {
                "type": ModelType.DETECTION,
                "accuracy": 92,
                "speed": 25,
                "memory": 1400,
                "size": 121.5,
                "hardware_tier": HardwareTier.HIGH_END
            },
            # Segmentation models
            "yolov8n-seg": {
                "type": ModelType.SEGMENTATION,
                "accuracy": 60,
                "speed": 90,
                "memory": 150,
                "size": 6.7,
                "hardware_tier": HardwareTier.LOW_END
            },
            "yolov8s-seg": {
                "type": ModelType.SEGMENTATION,
                "accuracy": 70,
                "speed": 80,
                "memory": 300,
                "size": 22.7,
                "hardware_tier": HardwareTier.LOW_END
            },
            "yolov8m-seg": {
                "type": ModelType.SEGMENTATION,
                "accuracy": 78,
                "speed": 60,
                "memory": 600,
                "size": 54.2,
                "hardware_tier": HardwareTier.MID_RANGE
            },
            "yolov8l-seg": {
                "type": ModelType.SEGMENTATION,
                "accuracy": 82,
                "speed": 40,
                "memory": 900,
                "size": 95.7,
                "hardware_tier": HardwareTier.HIGH_END
            },
            "yolov8x-seg": {
                "type": ModelType.SEGMENTATION,
                "accuracy": 85,
                "speed": 25,
                "memory": 1300,
                "size": 166.8,
                "hardware_tier": HardwareTier.HIGH_END
            },
            "sam-vit-base": {
                "type": ModelType.SAM,
                "accuracy": 85,
                "speed": 40,
                "memory": 1300,
                "size": 358,
                "hardware_tier": HardwareTier.HIGH_END
            },
            "sam-vit-large": {
                "type": ModelType.SAM,
                "accuracy": 88,
                "speed": 30,
                "memory": 2500,
                "size": 929,
                "hardware_tier": HardwareTier.HIGH_END
            },
            "sam-vit-huge": {
                "type": ModelType.SAM,
                "accuracy": 90,
                "speed": 20,
                "memory": 4000,
                "size": 1772,
                "hardware_tier": HardwareTier.HIGH_END
            },
            # Pose models
            "yolov8n-pose": {
                "type": ModelType.POSE,
                "accuracy": 65,
                "speed": 90,
                "memory": 130,
                "size": 3.3,
                "hardware_tier": HardwareTier.LOW_END
            },
            "yolov8s-pose": {
                "type": ModelType.POSE,
                "accuracy": 75,
                "speed": 80,
                "memory": 280,
                "size": 11.7,
                "hardware_tier": HardwareTier.LOW_END
            },
            "yolov8m-pose": {
                "type": ModelType.POSE,
                "accuracy": 82,
                "speed": 60,
                "memory": 550,
                "size": 27.3,
                "hardware_tier": HardwareTier.MID_RANGE
            },
            "yolov8l-pose": {
                "type": ModelType.POSE,
                "accuracy": 86,
                "speed": 40,
                "memory": 850,
                "size": 48.6,
                "hardware_tier": HardwareTier.HIGH_END
            },
            "yolov8x-pose": {
                "type": ModelType.POSE,
                "accuracy": 89,
                "speed": 25,
                "memory": 1250,
                "size": 89.6,
                "hardware_tier": HardwareTier.HIGH_END
            },
            # CLIP models
            "clip-vit-base-patch32": {
                "type": ModelType.CLIP,
                "accuracy": 85,
                "speed": 50,
                "memory": 1000,
                "size": 354,
                "hardware_tier": HardwareTier.MID_RANGE
            },
            "clip-vit-base-patch16": {
                "type": ModelType.CLIP,
                "accuracy": 87,
                "speed": 40,
                "memory": 1200,
                "size": 354,
                "hardware_tier": HardwareTier.MID_RANGE
            },
            "clip-vit-large-patch14": {
                "type": ModelType.CLIP,
                "accuracy": 90,
                "speed": 25,
                "memory": 3000,
                "size": 1763,
                "hardware_tier": HardwareTier.HIGH_END
            },
            # DETR models
            "detr-resnet50": {
                "type": ModelType.DETR,
                "accuracy": 75,
                "speed": 30,
                "memory": 1500,
                "size": 167,
                "hardware_tier": HardwareTier.HIGH_END
            },
            "detr-resnet101": {
                "type": ModelType.DETR,
                "accuracy": 78,
                "speed": 20,
                "memory": 2500,
                "size": 275,
                "hardware_tier": HardwareTier.HIGH_END
            },
            "rfdetr-resnet50": {
                "type": ModelType.RFDETR,
                "accuracy": 80,
                "speed": 40,
                "memory": 1800,
                "size": 200,
                "hardware_tier": HardwareTier.HIGH_END
            },
            "rfdetr-resnet101": {
                "type": ModelType.RFDETR,
                "accuracy": 83,
                "speed": 30,
                "memory": 2800,
                "size": 308,
                "hardware_tier": HardwareTier.HIGH_END
            },
            # ReID models
            "reid-resnet50": {
                "type": ModelType.REID,
                "accuracy": 85,
                "speed": 60,
                "memory": 500,
                "size": 97,
                "hardware_tier": HardwareTier.MID_RANGE
            },
            "reid-resnet101": {
                "type": ModelType.REID,
                "accuracy": 88,
                "speed": 45,
                "memory": 800,
                "size": 178,
                "hardware_tier": HardwareTier.MID_RANGE
            },
            "reid-ibn-resnet50": {
                "type": ModelType.REID,
                "accuracy": 87,
                "speed": 55,
                "memory": 550,
                "size": 105,
                "hardware_tier": HardwareTier.MID_RANGE
            },
            # Face models
            "arcface-r50": {
                "type": ModelType.FACE,
                "accuracy": 90,
                "speed": 70,
                "memory": 250,
                "size": 97,
                "hardware_tier": HardwareTier.LOW_END
            },
            "arcface-r100": {
                "type": ModelType.FACE,
                "accuracy": 93,
                "speed": 50,
                "memory": 450,
                "size": 178,
                "hardware_tier": HardwareTier.MID_RANGE
            }
        }
    
    def _detect_hardware(self) -> HardwareInfo:
        """
        Detect hardware information
        
        Returns:
            HardwareInfo object with detected hardware information
        """
        # Get platform
        current_platform = platform.system()
        
        # Get CPU cores
        cpu_cores = os.cpu_count() or 1
        
        # Get CPU RAM
        if current_platform == "Windows":
            try:
                import psutil
                cpu_ram = int(psutil.virtual_memory().total / (1024 * 1024))
            except ImportError:
                # Fallback to Windows command
                output = subprocess.check_output(["wmic", "OS", "get", "TotalVisibleMemorySize"], universal_newlines=True)
                memory_kb = int(output.strip().split("\n")[1])
                cpu_ram = int(memory_kb / 1024)
        elif current_platform == "Linux":
            try:
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            memory_kb = int(line.split()[1])
                            cpu_ram = int(memory_kb / 1024)
                            break
            except Exception:
                cpu_ram = 4096  # Default to 4GB
        elif current_platform == "Darwin":
            try:
                output = subprocess.check_output(["sysctl", "hw.memsize"], universal_newlines=True)
                memory_bytes = int(output.split(":")[1].strip())
                cpu_ram = int(memory_bytes / (1024 * 1024))
            except Exception:
                cpu_ram = 4096  # Default to 4GB
        else:
            cpu_ram = 4096  # Default to 4GB
        
        # Check GPU availability
        gpu_available = False
        gpu_memory = 0
        gpu_name = ""
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_memory = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
                gpu_name = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        
        # Determine hardware tier
        hardware_tier = self._calculate_hardware_tier(cpu_cores, cpu_ram, gpu_available, gpu_memory)
        
        return HardwareInfo(
            platform=current_platform,
            cpu_cores=cpu_cores,
            cpu_ram=cpu_ram,
            gpu_available=gpu_available,
            gpu_memory=gpu_memory,
            gpu_name=gpu_name,
            hardware_tier=hardware_tier
        )
    
    def _calculate_hardware_tier(self, cpu_cores: int, cpu_ram: int, gpu_available: bool, gpu_memory: int) -> HardwareTier:
        """
        Calculate hardware tier based on hardware specifications
        
        Args:
            cpu_cores: Number of CPU cores
            cpu_ram: CPU RAM in MB
            gpu_available: Whether GPU is available
            gpu_memory: GPU memory in MB
        
        Returns:
            HardwareTier enum value
        """
        if gpu_available:
            if gpu_memory > 8000:
                return HardwareTier.HIGH_END
            elif gpu_memory > 4000:
                return HardwareTier.MID_RANGE
            elif gpu_memory > 2000:
                return HardwareTier.LOW_END
            else:
                return HardwareTier.EDGE
        else:
            if cpu_ram > 16000 and cpu_cores > 8:
                return HardwareTier.MID_RANGE
            elif cpu_ram > 8000 and cpu_cores > 4:
                return HardwareTier.LOW_END
            elif cpu_ram > 4000:
                return HardwareTier.EDGE
            else:
                return HardwareTier.MOBILE
    
    def select_model(self, requirement: ModelRequirement) -> Dict[str, Any]:
        """
        Select the most appropriate model based on requirements
        
        Args:
            requirement: Model requirement
        
        Returns:
            Dictionary with selected model information
        """
        # Filter models by type
        filtered_models = []
        for model_name, model_info in self._model_database.items():
            if model_info["type"] == requirement.model_type:
                # Check if model meets memory requirement
                if model_info["memory"] <= requirement.memory:
                    # Check if model is compatible with hardware tier
                    if self._is_compatible_with_hardware(model_info["hardware_tier"]):
                        filtered_models.append((model_name, model_info))
        
        if not filtered_models:
            # If no models meet the requirements, return the smallest model of the required type
            fallback_models = []
            for model_name, model_info in self._model_database.items():
                if model_info["type"] == requirement.model_type:
                    fallback_models.append((model_name, model_info))
            
            if fallback_models:
                # Sort by memory usage
                fallback_models.sort(key=lambda x: x[1]["memory"])
                selected_model = fallback_models[0]
                return {
                    "model_name": selected_model[0],
                    "model_info": selected_model[1],
                    "reason": "No models met memory requirements, selected smallest model",
                    "hardware_info": self._hardware_info
                }
            else:
                return {
                    "error": "No models found for the specified type",
                    "hardware_info": self._hardware_info
                }
        
        # Calculate score for each model
        scored_models = []
        for model_name, model_info in filtered_models:
            # Calculate weighted score
            accuracy_weight = 0.4
            speed_weight = 0.3
            memory_weight = 0.3
            
            # Normalize memory usage (lower is better)
            memory_score = max(0, 100 - (model_info["memory"] / requirement.memory) * 100)
            
            # Calculate total score
            score = (
                (model_info["accuracy"] / 100) * accuracy_weight +
                (model_info["speed"] / 100) * speed_weight +
                (memory_score / 100) * memory_weight
            ) * 100
            
            scored_models.append((model_name, model_info, score))
        
        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[2], reverse=True)
        
        # Select top model
        selected_model = scored_models[0]
        
        return {
            "model_name": selected_model[0],
            "model_info": selected_model[1],
            "score": selected_model[2],
            "reason": "Selected based on weighted score",
            "hardware_info": self._hardware_info
        }
    
    def _is_compatible_with_hardware(self, model_tier: HardwareTier) -> bool:
        """
        Check if model is compatible with detected hardware
        
        Args:
            model_tier: Hardware tier required by the model
        
        Returns:
            True if compatible, False otherwise
        """
        # Hardware tier compatibility mapping
        compatibility = {
            HardwareTier.HIGH_END: [HardwareTier.HIGH_END],
            HardwareTier.MID_RANGE: [HardwareTier.HIGH_END, HardwareTier.MID_RANGE],
            HardwareTier.LOW_END: [HardwareTier.HIGH_END, HardwareTier.MID_RANGE, HardwareTier.LOW_END],
            HardwareTier.EDGE: [HardwareTier.HIGH_END, HardwareTier.MID_RANGE, HardwareTier.LOW_END, HardwareTier.EDGE],
            HardwareTier.MOBILE: [HardwareTier.HIGH_END, HardwareTier.MID_RANGE, HardwareTier.LOW_END, HardwareTier.EDGE, HardwareTier.MOBILE]
        }
        
        return self._hardware_info.hardware_tier in compatibility.get(model_tier, [])
    
    def get_hardware_info(self) -> HardwareInfo:
        """
        Get detected hardware information
        
        Returns:
            HardwareInfo object
        """
        return self._hardware_info
    
    def get_available_models(self, model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """
        Get list of available models
        
        Args:
            model_type: Filter by model type
        
        Returns:
            List of available models
        """
        available_models = []
        for model_name, model_info in self._model_database.items():
            if model_type is None or model_info["type"] == model_type:
                if self._is_compatible_with_hardware(model_info["hardware_tier"]):
                    available_models.append({
                        "name": model_name,
                        "type": model_info["type"],
                        "accuracy": model_info["accuracy"],
                        "speed": model_info["speed"],
                        "memory": model_info["memory"],
                        "size": model_info["size"],
                        "hardware_tier": model_info["hardware_tier"]
                    })
        
        return available_models


# Global model selector instance
_model_selector: Optional[ModelSelector] = None


def get_model_selector() -> ModelSelector:
    """
    Get or create global model selector instance
    
    Returns:
        ModelSelector instance
    """
    global _model_selector
    if _model_selector is None:
        _model_selector = ModelSelector()
    return _model_selector


def select_model(
    model_type: Union[str, ModelType],
    accuracy: int = 70,
    speed: int = 50,
    memory: int = 1024,
    task: str = "general",
    constraints: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Select model based on requirements
    
    Args:
        model_type: Model type
        accuracy: Required accuracy level (0-100)
        speed: Required speed level (0-100)
        memory: Required memory in MB
        task: Specific task description
        constraints: Additional constraints
    
    Returns:
        Dictionary with selected model information
    """
    # Convert model_type string to ModelType enum
    if isinstance(model_type, str):
        model_type = ModelType(model_type)
    
    # Create requirement
    requirement = ModelRequirement(
        model_type=model_type,
        accuracy=accuracy,
        speed=speed,
        memory=memory,
        task=task,
        constraints=constraints
    )
    
    # Get model selector
    selector = get_model_selector()
    
    # Select model
    return selector.select_model(requirement)


def get_hardware_info() -> HardwareInfo:
    """
    Get hardware information
    
    Returns:
        HardwareInfo object
    """
    selector = get_model_selector()
    return selector.get_hardware_info()


def get_available_models(model_type: Optional[Union[str, ModelType]] = None) -> List[Dict[str, Any]]:
    """
    Get available models
    
    Args:
        model_type: Filter by model type
    
    Returns:
        List of available models
    """
    # Convert model_type string to ModelType enum
    if isinstance(model_type, str):
        model_type = ModelType(model_type)
    
    selector = get_model_selector()
    return selector.get_available_models(model_type)
