"""
Pydantic Configuration Models and Manager

This module defines Pydantic models for configuration validation and
a configuration manager for loading/saving configurations from files.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, field_validator
import json

# Try to import yaml, but make it optional
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


class BaseConfig(BaseModel):
    """Base configuration model with common validation"""
    class Config:
        extra = "allow"  # Allow additional fields not explicitly defined
        from_attributes = True  # Support loading from objects
        arbitrary_types_allowed = True


class DetectorConfig(BaseConfig):
    """Detector configuration model"""
    model_path: str = Field(default="yolov8n.pt", description="Path to model file")
    model_type: str = Field(default="yolo", description="Type of model (yolo, detr, rfdetr)")
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Confidence threshold")
    category_thresholds: Optional[Dict[str, float]] = Field(default=None, description="Category-specific confidence thresholds")
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")
    device: str = Field(default="cpu", description="Device to use for inference")
    enable_segmentation: bool = Field(default=False, description="Enable segmentation for YOLO")
    batch_inference: bool = Field(default=False, description="Enable batch inference")
    dynamic_batch_size: bool = Field(default=False, description="Enable dynamic batch size")
    max_batch_size: int = Field(default=8, ge=1, description="Maximum batch size for dynamic batching")
    min_batch_size: int = Field(default=1, ge=1, description="Minimum batch size for dynamic batching")
    use_fp16: bool = Field(default=False, description="Use FP16 precision for inference")
    detr_model_name: str = Field(default="facebook/detr-resnet-50", description="DETR model name from HuggingFace")
    rfdetr_model_name: Optional[str] = Field(default=None, description="RF-DETR model name")
    
    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        """Validate model type is supported"""
        supported_types = ["yolo", "detr", "rfdetr"]
        if v not in supported_types:
            raise ValueError(f"Invalid model_type: {v}. Supported: {', '.join(supported_types)}")
        return v
    
    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device is supported"""
        supported_devices = ["cpu", "cuda", "mps"]
        v_lower = v.lower()
        if v_lower not in supported_devices:
            return "cpu"  # Default to CPU if invalid
        return v_lower
    
    @field_validator("category_thresholds")
    @classmethod
    def validate_category_thresholds(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """Validate category thresholds are between 0.0 and 1.0"""
        if v is None:
            return v
        for category, threshold in v.items():
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Threshold for category '{category}' must be between 0.0 and 1.0, got {threshold}")
        return v


class TrackerConfig(BaseConfig):
    """Tracker configuration model"""
    max_age: int = Field(default=30, ge=1, description="Maximum frames to keep a track alive without detection")
    min_hits: int = Field(default=3, ge=1, description="Minimum detections to start a new track")
    iou_threshold: float = Field(default=0.3, ge=0.0, le=1.0, description="IoU threshold for track matching")
    use_kalman: bool = Field(default=False, description="Use Kalman filtering for tracking")
    track_history_length: int = Field(default=30, ge=1, description="Length of track history to maintain")
    new_track_activation_conf: float = Field(default=0.6, ge=0.0, le=1.0, description="Confidence threshold for new track activation")
    embedding_dim: int = Field(default=2048, ge=1, description="Embedding dimension for re-identification")
    matching_strategy: str = Field(default="hungarian", description="Matching strategy (hungarian or greedy)")
    matching_cost_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Matching cost threshold")
    
    @field_validator("matching_strategy")
    @classmethod
    def validate_matching_strategy(cls, v: str) -> str:
        """Validate matching strategy is supported"""
        supported_strategies = ["hungarian", "greedy"]
        if v not in supported_strategies:
            raise ValueError(f"Invalid matching_strategy: {v}. Supported: {', '.join(supported_strategies)}")
        return v


class PerformanceConfig(BaseConfig):
    """Performance configuration model"""
    batch_inference: bool = Field(default=False, description="Enable batch inference")
    use_fp16: bool = Field(default=False, description="Use FP16 precision")
    video_async_read: bool = Field(default=False, description="Enable asynchronous video reading")


class PipelineConfig(BaseConfig):
    """Vision pipeline configuration model"""
    enable_tracking: bool = Field(default=True, description="Enable tracking in pipeline")
    detector_config: DetectorConfig = Field(default_factory=DetectorConfig, description="Detector configuration")
    tracker_config: TrackerConfig = Field(default_factory=TrackerConfig, description="Tracker configuration")
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig, description="Performance configuration")


class VisualizerConfig(BaseConfig):
    """Visualizer configuration model"""
    show_labels: bool = Field(default=True, description="Show class labels")
    show_confidences: bool = Field(default=True, description="Show confidence scores")
    show_track_ids: bool = Field(default=True, description="Show track IDs")
    line_thickness: int = Field(default=2, ge=1, description="Line thickness for drawing")
    font_scale: float = Field(default=0.5, ge=0.1, description="Font scale for text")


class AutoLabelerConfig(BaseConfig):
    """Auto labeler configuration model"""
    detector_config: DetectorConfig = Field(default_factory=DetectorConfig, description="Detector configuration")
    tracker_config: Optional[TrackerConfig] = Field(default=None, description="Tracker configuration (if tracking enabled)")
    enable_tracking: bool = Field(default=False, description="Enable tracking for video labeling")
    output_format: str = Field(default="coco", description="Output format (json, csv, coco)")
    output_path: str = Field(default="output/annotations", description="Output directory path")
    
    @field_validator("output_format")
    @classmethod
    def validate_output_format(cls, v: str) -> str:
        """Validate output format is supported"""
        supported_formats = ["json", "csv", "coco"]
        if v not in supported_formats:
            raise ValueError(f"Invalid output_format: {v}. Supported: {', '.join(supported_formats)}")
        return v


class DeviceManager:
    """
    Utility to validate and normalize device strings.
    
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
    def normalize_device(device: Optional[str]) -> str:
        """
        Normalize device string to one of 'cpu', 'cuda', 'mps'.
        
        If the requested device is unavailable, falls back to 'cpu'.
        """
        if not device or device.lower() == "auto":
            # Auto-select best available device
            if DeviceManager.is_cuda_available():
                return "cuda"
            elif DeviceManager.is_mps_available():
                return "mps"
            return "cpu"

        d = device.lower()
        
        # Handle CUDA device specification
        if d.startswith("cuda"):
            return "cuda" if DeviceManager.is_cuda_available() else "cpu"
        
        # Handle MPS
        if d == "mps":
            return "mps" if DeviceManager.is_mps_available() else "cpu"
        
        # Handle CPU
        if d == "cpu":
            return "cpu"
        
        # Invalid device string, auto-select best available
        return "cuda" if DeviceManager.is_cuda_available() else "mps" if DeviceManager.is_mps_available() else "cpu"


class Config:
    """
    Configuration manager for loading/saving configurations
    
    This class provides utilities for loading and saving configurations
    from files. It supports both JSON and YAML formats.
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
        """
        from pathlib import Path
        
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
                f"Unsupported file format: {suffix}. Supported: .json, .yaml, .yml"
            )
    
    @staticmethod
    def save_to_file(config: Dict[str, Any], file_path: str, format: Optional[str] = None):
        """
        Save configuration to file (supports JSON and YAML)
        
        Args:
            config: Configuration dictionary to save
            file_path: Output file path
            format: Optional format specification ('json' or 'yaml')
        """
        from pathlib import Path
        
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
    
    @classmethod
    def load_as_model(cls, file_path: str, model_type):
        """
        Load configuration from file and parse it into a Pydantic model
        
        Args:
            file_path: Path to configuration file
            model_type: Pydantic model class to use for parsing
        
        Returns:
            Parsed Pydantic model instance
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
    
    # Default configuration methods
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
        from ...models import get_model_manager
        
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
