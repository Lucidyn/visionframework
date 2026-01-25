"""
Configuration utilities

This module provides configuration loading and management utilities,
supporting both JSON and YAML formats.
"""

import json
from typing import Dict, Any, Optional, Type, TypeVar
from pathlib import Path

# Import pydantic models
from .config_models import (
    DetectorConfig, TrackerConfig, PipelineConfig,
    VisualizerConfig, BaseConfig, PerformanceConfig
)

# Try to import yaml, but make it optional
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

# Type variable for generic methods
T = TypeVar('T', bound=BaseConfig)


class Config:
    """
    Configuration manager
    
    This class provides utilities for loading and saving configurations
    from files. It supports both JSON and YAML formats.
    
    Example:
        ```python
        # Load from YAML
        config = Config.load_from_file("config.yaml")
        
        # Load from JSON
        config = Config.load_from_file("config.json")
        
        # Get default configurations
        detector_config = Config.get_default_detector_config()
        ```
    """
    
    @staticmethod
    def load_from_file(file_path: str) -> Dict[str, Any]:
        """
        Load configuration from file (supports JSON and YAML)
        
        Args:
            file_path: Path to configuration file (.json, .yaml, or .yml)
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        
        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is not supported
            ImportError: If YAML file is used but PyYAML is not installed
        
        Note:
            File format is determined by extension:
            - .json: JSON format (built-in support)
            - .yaml, .yml: YAML format (requires PyYAML)
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        suffix = path.suffix.lower()
        
        if suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif suffix in ['.yaml', '.yml']:
            if not YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is required for YAML files. Install with: pip install pyyaml"
                )
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. "
                "Supported formats: .json, .yaml, .yml"
            )
    
    @staticmethod
    def save_to_file(config: Dict[str, Any], file_path: str, format: Optional[str] = None):
        """
        Save configuration to file (supports JSON and YAML)
        
        Args:
            config: Configuration dictionary to save
            file_path: Output file path
            format: Optional format specification ('json' or 'yaml').
                   If None, format is determined by file extension.
        
        Raises:
            ValueError: If format is not supported
            ImportError: If YAML format is used but PyYAML is not installed
        """
        path = Path(file_path)
        
        # Determine format
        if format is None:
            suffix = path.suffix.lower()
            if suffix == '.json':
                format = 'json'
            elif suffix in ['.yaml', '.yml']:
                format = 'yaml'
            else:
                format = 'json'  # Default to JSON
        
        if format == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        elif format == 'yaml':
            if not YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is required for YAML files. Install with: pip install pyyaml"
                )
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Supported: 'json', 'yaml'")
    
    @staticmethod
    def get_default_detector_config() -> Dict[str, Any]:
        """Get default detector configuration"""
        return DetectorConfig().model_dump()
    
    @staticmethod
    def get_default_tracker_config() -> Dict[str, Any]:
        """Get default tracker configuration"""
        return TrackerConfig().model_dump()
    
    @staticmethod
    def get_default_pipeline_config() -> Dict[str, Any]:
        """Get default pipeline configuration"""
        return PipelineConfig().model_dump()
    
    @staticmethod
    def get_default_visualizer_config() -> Dict[str, Any]:
        """Get default visualizer configuration"""
        return VisualizerConfig().model_dump()
    
    @staticmethod
    def get_default_performance_config() -> Dict[str, Any]:
        """Get default performance configuration"""
        return PerformanceConfig().model_dump()
    
    @classmethod
    def load_as_model(cls, file_path: str, model_type: Type[T]) -> T:
        """
        Load configuration from file and parse it into a Pydantic model
        
        Args:
            file_path: Path to configuration file
            model_type: Pydantic model class to use for parsing
        
        Returns:
            T: Parsed Pydantic model instance
        """
        config_dict = cls.load_from_file(file_path)
        return model_type(**config_dict)
    
    @classmethod
    def save_model(cls, model: BaseConfig, file_path: str, format: Optional[str] = None):
        """
        Save a Pydantic model to a configuration file
        
        Args:
            model: Pydantic model instance to save
            file_path: Output file path
            format: Optional format specification ('json' or 'yaml')
        """
        config_dict = model.model_dump()
        cls.save_to_file(config_dict, file_path, format)


class DeviceManager:
    """Utility to validate and normalize device strings.

    Provides a centralized way to interpret device strings like 'cpu', 'cuda',
    and 'mps'. It will fall back to 'cpu' if the requested accelerator is
    unavailable.
    """

    @staticmethod
    def is_cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def is_mps_available() -> bool:
        try:
            import torch
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        except Exception:
            return False
    
    @staticmethod
    def get_cuda_device_count() -> int:
        """Get the number of available CUDA devices."""
        try:
            import torch
            if DeviceManager.is_cuda_available():
                return torch.cuda.device_count()
        except Exception:
            pass
        return 0
    
    @staticmethod
    def get_cuda_device_info(device_id: int = 0) -> Optional[dict]:
        """Get detailed information about a CUDA device.
        
        Args:
            device_id: Index of the CUDA device to query
            
        Returns:
            Dictionary with device information, or None if unavailable
        """
        try:
            import torch
            if not DeviceManager.is_cuda_available():
                return None
            
            if device_id >= torch.cuda.device_count():
                return None
            
            device = torch.cuda.get_device_properties(device_id)
            return {
                "name": device.name,
                "memory_total": device.total_memory,
                "memory_available": torch.cuda.get_device_properties(device_id).total_memory - torch.cuda.memory_allocated(device_id),
                "compute_capability": (device.major, device.minor),
                "device_id": device_id
            }
        except Exception:
            return None
    
    @staticmethod
    def get_available_devices() -> list:
        """Get a list of all available devices.
        
        Returns:
            List of strings representing available devices
        """
        devices = ["cpu"]
        
        if DeviceManager.is_cuda_available():
            device_count = DeviceManager.get_cuda_device_count()
            for i in range(device_count):
                devices.append(f"cuda:{i}")
            devices.append("cuda")  # Default CUDA device
        
        if DeviceManager.is_mps_available():
            devices.append("mps")
        
        return devices
    
    @staticmethod
    def auto_select_device(priority: list = None) -> str:
        """Automatically select the best available device based on priority.
        
        Args:
            priority: List of device types in order of preference (default: ["cuda", "mps", "cpu"])
            
        Returns:
            String representing the best available device
        """
        if priority is None:
            priority = ["cuda", "mps", "cpu"]
        
        for device_type in priority:
            if device_type == "cuda" and DeviceManager.is_cuda_available():
                return "cuda"
            elif device_type == "mps" and DeviceManager.is_mps_available():
                return "mps"
            elif device_type == "cpu":
                return "cpu"
        
        return "cpu"  # Fallback to CPU

    @staticmethod
    def normalize_device(device: Optional[str]) -> str:
        """Normalize device string to one of 'cpu', 'cuda', 'mps' or 'cuda:{id}'.

        If the requested device is unavailable, falls back to 'cpu'.
        """
        if not device or device.lower() == "auto":
            return DeviceManager.auto_select_device()

        d = device.lower()
        
        # Handle CUDA device specification (cuda, cuda:0, cuda:1, etc.)
        if d.startswith("cuda"):
            if DeviceManager.is_cuda_available():
                if d == "cuda":
                    return "cuda"  # Use default CUDA device
                elif ":" in d:
                    # Check if specified CUDA device exists
                    try:
                        device_id = int(d.split(":")[1])
                        if device_id < DeviceManager.get_cuda_device_count():
                            return d
                    except Exception:
                        pass
                # Fall back to default CUDA device if specified one is invalid
                return "cuda"
            return "cpu"
        
        # Handle MPS
        if d == "mps":
            return "mps" if DeviceManager.is_mps_available() else "cpu"
        
        # Handle CPU
        if d == "cpu":
            return "cpu"
        
        # Invalid device string, auto-select best available
        return DeviceManager.auto_select_device()
    
    @staticmethod
    def get_device_info(device: str) -> dict:
        """Get detailed information about a specific device.
        
        Args:
            device: Device string to get information about
            
        Returns:
            Dictionary with device information
        """
        normalized_device = DeviceManager.normalize_device(device)
        
        if normalized_device.startswith("cuda"):
            # Get CUDA device info
            device_id = 0
            if ":" in normalized_device:
                device_id = int(normalized_device.split(":")[1])
            cuda_info = DeviceManager.get_cuda_device_info(device_id)
            if cuda_info:
                return {
                    "type": "cuda",
                    "available": True,
                    "normalized": normalized_device,
                    **cuda_info
                }
        elif normalized_device == "mps":
            # Get MPS device info
            return {
                "type": "mps",
                "available": True,
                "normalized": normalized_device,
                "name": "Apple Silicon GPU"
            }
        elif normalized_device == "cpu":
            # Get CPU device info
            try:
                import platform
                import psutil
                return {
                    "type": "cpu",
                    "available": True,
                    "normalized": normalized_device,
                    "name": platform.processor(),
                    "cpu_count": psutil.cpu_count(logical=True),
                    "memory_total": psutil.virtual_memory().total
                }
            except Exception:
                return {
                    "type": "cpu",
                    "available": True,
                    "normalized": normalized_device,
                    "name": "Unknown CPU"
                }
        
        return {
            "type": "unknown",
            "available": False,
            "normalized": normalized_device
        }

class ModelCache:
    """Simple in-memory model cache with reference counting.

    Stores loaded model instances keyed by a string (typically model path).
    Use `get_model(key, loader)` to obtain a cached model (loader called on first load),
    and `release_model(key)` to decrement reference count and free the model when unused.
    """

    _cache = {}
    _lock = None

    try:
        import threading
        _lock = threading.Lock()
    except Exception:
        _lock = None

    @classmethod
    def get_model(cls, key: str, loader):
        """Return cached model for `key`. `loader` is a callable to create the model if missing."""
        if cls._lock:
            cls._lock.acquire()
        try:
            entry = cls._cache.get(key)
            if entry is not None:
                model, ref = entry
                cls._cache[key] = (model, ref + 1)
                return model

            # load model
            model = loader()
            cls._cache[key] = (model, 1)
            return model
        finally:
            if cls._lock:
                cls._lock.release()

    @classmethod
    def release_model(cls, key: str):
        """Release a reference to the cached model and free if refcount reaches zero."""
        if cls._lock:
            cls._lock.acquire()
        try:
            entry = cls._cache.get(key)
            if not entry:
                return
            model, ref = entry
            ref -= 1
            if ref <= 0:
                # Attempt graceful cleanup
                try:
                    if hasattr(model, 'to'):
                        try:
                            model.to('cpu')
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    del model
                except Exception:
                    pass
                cls._cache.pop(key, None)
                # Try to free CUDA cache
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                cls._cache[key] = (model, ref)
        finally:
            if cls._lock:
                cls._lock.release()

    @classmethod
    def get_from_manager(cls, model_name: str, download: bool = True, verify_hash: bool = True):
        """Get model from ModelManager and cache it in memory.
        
        Args:
            model_name: Name of the model to get
            download: Whether to download if not cached
            verify_hash: Whether to verify file hash after download
            
        Returns:
            Model instance or None if not found
        """
        from ..models import get_model_manager
        
        def loader():
            model_manager = get_model_manager()
            model_path = model_manager.get_model_path(model_name, download=download, verify_hash=verify_hash)
            if model_path:
                # Try to import and load YOLO model
                try:
                    from ultralytics import YOLO
                    return YOLO(str(model_path))
                except Exception:
                    pass
            return None
        
        return cls.get_model(model_name, loader)


