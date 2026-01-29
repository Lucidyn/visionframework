"""
Pydantic Configuration Models and Manager

This module defines Pydantic models for configuration validation and
a configuration manager for loading/saving configurations from files.
"""

from typing import Dict, Any, Optional, List, Union, Callable
from pydantic import BaseModel, Field, field_validator, ConfigDict
import json

# Try to import yaml, but make it optional
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


class BaseConfig(BaseModel):
    """Base configuration model with common validation and inheritance support"""
    model_config = ConfigDict(
        extra="allow",  # Allow additional fields not explicitly defined
        from_attributes=True,  # Support loading from objects
        arbitrary_types_allowed=True
    )
    
    @classmethod
    def from_parent(cls, parent_config: Optional['BaseConfig'] = None, **overrides) -> 'BaseConfig':
        """
        Create a new configuration by inheriting from a parent configuration
        
        Args:
            parent_config: Parent configuration to inherit from
            **overrides: Configuration overrides
            
        Returns:
            BaseConfig: New configuration with inherited values and overrides
        """
        # Start with empty dict
        config_dict = {}
        
        # Inherit from parent if provided
        if parent_config:
            config_dict.update(parent_config.model_dump())
        
        # Apply overrides
        config_dict.update(overrides)
        
        # Create new instance
        return cls(**config_dict)
    
    def merge(self, other_config: 'BaseConfig') -> 'BaseConfig':
        """
        Merge another configuration into this one
        
        Args:
            other_config: Configuration to merge
            
        Returns:
            BaseConfig: New configuration with merged values
        """
        merged_dict = self.model_dump()
        merged_dict.update(other_config.model_dump())
        return self.__class__(**merged_dict)
    
    def validate_config(self) -> bool:
        """
        Validate the configuration
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            # Pydantic already validates on creation, but we can add additional validation here
            return True
        except Exception:
            return False
    
    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """
        Get a nested configuration value using dot notation
        
        Args:
            key_path: Dot-separated key path (e.g., "detector_config.conf_threshold")
            default: Default value if key not found
            
        Returns:
            Any: The value at the specified path
        """
        keys = key_path.split('.')
        value = self.model_dump()
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set_nested(self, key_path: str, value: Any) -> 'BaseConfig':
        """
        Set a nested configuration value using dot notation
        
        Args:
            key_path: Dot-separated key path (e.g., "detector_config.conf_threshold")
            value: Value to set
            
        Returns:
            BaseConfig: New configuration with updated value
        """
        config_dict = self.model_dump()
        keys = key_path.split('.')
        
        # Traverse to the parent of the target key
        current = config_dict
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the value
        current[keys[-1]] = value
        
        return self.__class__(**config_dict)
    
    def inherit_from(self, parent_config: 'BaseConfig') -> 'BaseConfig':
        """
        Create a new configuration by inheriting from another configuration
        
        Args:
            parent_config: Parent configuration to inherit from
            
        Returns:
            BaseConfig: New configuration with inherited values
        """
        return self.from_parent(parent_config)
    
    def with_overrides(self, **overrides) -> 'BaseConfig':
        """
        Create a new configuration with specified overrides
        
        Args:
            **overrides: Configuration overrides
            
        Returns:
            BaseConfig: New configuration with overrides
        """
        return self.from_parent(self, **overrides)
    
    def get_config_path(self) -> Optional[str]:
        """
        Get the path to the configuration file
        
        Returns:
            Optional[str]: Path to the configuration file if available
        """
        return getattr(self, '_config_path', None)
    
    def set_config_path(self, path: str) -> 'BaseConfig':
        """
        Set the path to the configuration file
        
        Args:
            path: Path to the configuration file
            
        Returns:
            BaseConfig: New configuration with updated path
        """
        config_dict = self.model_dump()
        new_config = self.__class__(**config_dict)
        new_config._config_path = path
        return new_config
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary
        
        Returns:
            Dict[str, Any]: Configuration as dictionary
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """
        Create configuration from dictionary
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            BaseConfig: Configuration instance
        """
        return cls(**config_dict)
    
    def validate_and_get(self, key: str, default: Any = None, validator: Optional[Callable] = None) -> Any:
        """
        Validate and get a configuration value
        
        Args:
            key: Configuration key
            default: Default value if key not found
            validator: Optional validation function
            
        Returns:
            Any: Validated configuration value
        """
        value = getattr(self, key, default)
        
        # Apply validation if provided
        if validator:
            try:
                if validator(value):
                    return value
                return default
            except Exception:
                return default
        
        return value


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
    from files. It supports both JSON and YAML formats, and includes
    support for configuration inheritance and advanced validation.
    """
    
    # Configuration version
    CONFIG_VERSION = "1.0"
    
    @staticmethod
    def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries
        
        Args:
            dict1: First dictionary (base)
            dict2: Second dictionary (overrides)
            
        Returns:
            Dict[str, Any]: Deep merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = Config._deep_merge(result[key], value)
            else:
                # Override with new value
                result[key] = value
        
        return result
    
    @staticmethod
    def load_with_inheritance(base_file: str, *override_files: str) -> Dict[str, Any]:
        """
        Load configuration with inheritance from multiple files
        
        Args:
            base_file: Base configuration file
            *override_files: Override configuration files
            
        Returns:
            Dict[str, Any]: Merged configuration dictionary
        """
        # Load base config
        config_dict = Config.load_from_file(base_file)
        
        # Apply overrides in order
        for override_file in override_files:
            override_dict = Config.load_from_file(override_file)
            config_dict = Config._deep_merge(config_dict, override_dict)
        
        # Remove config_version if present
        config_dict.pop('config_version', None)
        
        return config_dict
    
    @classmethod
    def load_model_with_inheritance(cls, base_file: str, model_type, *override_files: str) -> BaseConfig:
        """
        Load configuration with inheritance and parse into a Pydantic model
        
        Args:
            base_file: Base configuration file
            model_type: Pydantic model class
            *override_files: Override configuration files
            
        Returns:
            BaseConfig: Merged configuration model
        """
        config_dict = cls.load_with_inheritance(base_file, *override_files)
        model = model_type(**config_dict)
        model._config_path = base_file
        return model
    
    @staticmethod
    def compare_configs(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two configurations and return differences
        
        Args:
            config1: First configuration
            config2: Second configuration
            
        Returns:
            Dict[str, Any]: Differences between configurations
        """
        differences = {}
        
        # Check keys in config1 not in config2
        for key in config1:
            if key not in config2:
                differences[key] = {"type": "removed", "old_value": config1[key]}
            elif config1[key] != config2[key]:
                if isinstance(config1[key], dict) and isinstance(config2[key], dict):
                    # Recursively compare nested dictionaries
                    nested_diff = Config.compare_configs(config1[key], config2[key])
                    if nested_diff:
                        differences[key] = {"type": "modified", "changes": nested_diff}
                else:
                    differences[key] = {"type": "modified", "old_value": config1[key], "new_value": config2[key]}
        
        # Check keys in config2 not in config1
        for key in config2:
            if key not in config1:
                differences[key] = {"type": "added", "new_value": config2[key]}
        
        return differences
    
    @staticmethod
    def validate_config_structure(config: Dict[str, Any], expected_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration structure against expected structure
        
        Args:
            config: Configuration to validate
            expected_structure: Expected structure with type hints
            
        Returns:
            Dict[str, Any]: Validation results
        """
        validation_results = {
            "valid": True,
            "errors": []
        }
        
        def _validate_recursive(config_part: Any, expected_part: Any, path: str = ""):
            """Recursively validate configuration structure"""
            if isinstance(expected_part, dict):
                if not isinstance(config_part, dict):
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"{path}: Expected dict, got {type(config_part).__name__}")
                    return
                
                for key, expected_value in expected_part.items():
                    current_path = f"{path}.{key}" if path else key
                    if key not in config_part:
                        validation_results["errors"].append(f"{current_path}: Missing required field")
                    else:
                        _validate_recursive(config_part[key], expected_value, current_path)
            elif isinstance(expected_part, list):
                if not isinstance(config_part, list):
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"{path}: Expected list, got {type(config_part).__name__}")
            elif expected_part is not None:
                # Check type
                expected_type = type(expected_part)
                if not isinstance(config_part, expected_type):
                    validation_results["valid"] = False
                    validation_results["errors"].append(f"{path}: Expected {expected_type.__name__}, got {type(config_part).__name__}")
        
        _validate_recursive(config, expected_structure)
        return validation_results
    
    @staticmethod
    def load_from_file(file_path: str, return_default_if_not_found: bool = False, default_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Load configuration from file (supports JSON and YAML)
        
        Args:
            file_path: Path to configuration file (.json, .yaml, or .yml)
            return_default_if_not_found: If True, return default_config instead of raising FileNotFoundError
            default_config: Default configuration to return if file not found
        
        Returns:
            Dict[str, Any]: Configuration dictionary
        
        Raises:
            FileNotFoundError: If the file does not exist and return_default_if_not_found is False
            ValueError: If the file format is not supported
            ImportError: If YAML file is used but PyYAML is not installed
        """
        from pathlib import Path
        
        path = Path(file_path)
        if not path.exists():
            if return_default_if_not_found:
                return default_config or {}
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
        # Create parent directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format
        if format is None:
            suffix = path.suffix.lower()
            if suffix == '.json':
                format = 'json'
            elif suffix in ['.yaml', '.yml']:
                format = 'yaml'
            else:
                format = 'json'  # Default to JSON
        
        # Add configuration version
        config_with_version = config.copy()
        config_with_version['config_version'] = Config.CONFIG_VERSION
        
        if format == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_with_version, f, indent=4, ensure_ascii=False)
        elif format == 'yaml':
            if not YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is required for YAML files. Install with: pip install pyyaml"
                )
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_with_version, f, default_flow_style=False, allow_unicode=True, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}. Supported: 'json', 'yaml'")
    
    @classmethod
    def load_as_model(cls, file_path: str, model_type, return_default_if_not_found: bool = False):
        """
        Load configuration from file and parse it into a Pydantic model
        
        Args:
            file_path: Path to configuration file
            model_type: Pydantic model class to use for parsing
            return_default_if_not_found: If True, return default model instance instead of raising FileNotFoundError
        
        Returns:
            Parsed Pydantic model instance
        """
        try:
            config_dict = cls.load_from_file(file_path)
            # Remove config_version if present
            config_dict.pop('config_version', None)
            model = model_type(**config_dict)
            model._config_path = file_path
            return model
        except FileNotFoundError:
            if return_default_if_not_found:
                return model_type()
            raise
    
    @classmethod
    def create_config_chain(cls, base_config: Union[str, Dict[str, Any]], *override_configs: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a configuration chain from multiple sources
        
        Args:
            base_config: Base configuration (file path or dictionary)
            *override_configs: Override configurations (file paths or dictionaries)
            
        Returns:
            Dict[str, Any]: Merged configuration dictionary
        """
        # Load base config
        if isinstance(base_config, str):
            config_dict = cls.load_from_file(base_config)
        else:
            config_dict = base_config.copy()
        
        # Apply overrides
        for override in override_configs:
            if isinstance(override, str):
                override_dict = cls.load_from_file(override)
            else:
                override_dict = override.copy()
            
            config_dict = cls._deep_merge(config_dict, override_dict)
        
        # Remove config_version if present
        config_dict.pop('config_version', None)
        
        return config_dict
    
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
    def get_default_detector_config_model() -> DetectorConfig:
        """Get default detector configuration as Pydantic model"""
        return DetectorConfig()
    
    @staticmethod
    def get_default_tracker_config() -> Dict[str, Any]:
        """Get default tracker configuration"""
        return TrackerConfig().model_dump()
    
    @staticmethod
    def get_default_tracker_config_model() -> TrackerConfig:
        """Get default tracker configuration as Pydantic model"""
        return TrackerConfig()
    
    @staticmethod
    def get_default_pipeline_config() -> Dict[str, Any]:
        """Get default pipeline configuration"""
        return PipelineConfig().model_dump()
    
    @staticmethod
    def get_default_pipeline_config_model() -> PipelineConfig:
        """Get default pipeline configuration as Pydantic model"""
        return PipelineConfig()
    
    @staticmethod
    def get_default_visualizer_config() -> Dict[str, Any]:
        """Get default visualizer configuration"""
        return VisualizerConfig().model_dump()
    
    @staticmethod
    def get_default_visualizer_config_model() -> VisualizerConfig:
        """Get default visualizer configuration as Pydantic model"""
        return VisualizerConfig()
    
    @staticmethod
    def get_default_performance_config() -> Dict[str, Any]:
        """Get default performance configuration"""
        return PerformanceConfig().model_dump()
    
    @staticmethod
    def get_default_performance_config_model() -> PerformanceConfig:
        """Get default performance configuration as Pydantic model"""
        return PerformanceConfig()
    
    @staticmethod
    def validate_config(config: Dict[str, Any], config_type: str) -> bool:
        """
        Validate configuration dictionary
        
        Args:
            config: Configuration dictionary to validate
            config_type: Type of configuration ('detector', 'tracker', 'pipeline', 'visualizer', 'performance')
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        try:
            if config_type == 'detector':
                DetectorConfig(**config)
            elif config_type == 'tracker':
                TrackerConfig(**config)
            elif config_type == 'pipeline':
                PipelineConfig(**config)
            elif config_type == 'visualizer':
                VisualizerConfig(**config)
            elif config_type == 'performance':
                PerformanceConfig(**config)
            else:
                return False
            return True
        except Exception:
            return False
    
    @classmethod
    def get_config_schema(cls, config_type: str) -> Dict[str, Any]:
        """
        Get configuration schema for a specific config type
        
        Args:
            config_type: Type of configuration ('detector', 'tracker', 'pipeline', 'visualizer', 'performance')
            
        Returns:
            Dict[str, Any]: Configuration schema
        """
        if config_type == 'detector':
            return DetectorConfig.model_json_schema()
        elif config_type == 'tracker':
            return TrackerConfig.model_json_schema()
        elif config_type == 'pipeline':
            return PipelineConfig.model_json_schema()
        elif config_type == 'visualizer':
            return VisualizerConfig.model_json_schema()
        elif config_type == 'performance':
            return PerformanceConfig.model_json_schema()
        else:
            return {}


class ModelCache:
    """Enhanced in-memory model cache with reference counting, LRU eviction, and multi-model support.

    Stores loaded model instances keyed by a string (typically model path).
    Use `get_model(key, loader)` to obtain a cached model (loader called on first load),
    and `release_model(key)` to decrement reference count and free the model when unused.
    
    Features:
    - Reference counting to track model usage
    - LRU eviction when max cache size is reached
    - Support for multiple model types (YOLO, CLIP, etc.)
    - Graceful resource cleanup
    - CUDA memory management
    """

    _cache = {}  # key: (model, ref_count, last_used_timestamp)
    _max_cache_size = 10  # Maximum number of models to keep in cache
    _lock = None
    _model_loaders = {
        'yolo': lambda path: None,  # Will be implemented with proper YOLO loading when needed
        'clip': lambda path: None,  # Will be implemented with proper CLIP loading
        'detr': lambda path: None,  # Will be implemented with proper DETR loading
        'pose': lambda path: None,  # Will be implemented with proper pose estimation loading
    }
    _loaders_initialized = False  # Flag to track if model loaders have been initialized
    
    @classmethod
    def _initialize_model_loaders(cls):
        """Initialize model loaders for different model types."""
        try:
            from ultralytics import YOLO
            cls._model_loaders['yolo'] = lambda path: YOLO(str(path))
            cls._model_loaders['pose'] = lambda path: YOLO(str(path))
        except ImportError:
            pass
        
        try:
            from transformers import CLIPModel, CLIPProcessor, DetrForObjectDetection, DetrImageProcessor
            def load_clip(path):
                try:
                    model = CLIPModel.from_pretrained(path)
                    processor = CLIPProcessor.from_pretrained(path)
                    return (model, processor)
                except Exception:
                    return None
            
            def load_detr(path):
                try:
                    model = DetrForObjectDetection.from_pretrained(path)
                    processor = DetrImageProcessor.from_pretrained(path)
                    return (model, processor)
                except Exception:
                    return None
            
            cls._model_loaders['clip'] = load_clip
            cls._model_loaders['detr'] = load_detr
        except ImportError:
            pass

    try:
        import threading
        _lock = threading.Lock()
    except Exception:
        _lock = None

    @classmethod
    def set_max_cache_size(cls, max_size: int):
        """Set maximum cache size for model instances."""
        if cls._lock:
            cls._lock.acquire()
        try:
            cls._max_cache_size = max(1, max_size)
            # If current cache exceeds new max size, evict least recently used models
            if len(cls._cache) > cls._max_cache_size:
                # Sort by last used timestamp
                sorted_keys = sorted(
                    cls._cache.keys(), 
                    key=lambda k: cls._cache[k][2]  # last_used_timestamp
                )
                # Evict models until cache size is within limit
                models_to_evict = len(cls._cache) - cls._max_cache_size
                for key in sorted_keys[:models_to_evict]:
                    entry = cls._cache.get(key)
                    if entry and entry[1] <= 0:  # Only evict unused models
                        cls._cache.pop(key, None)
        finally:
            if cls._lock:
                cls._lock.release()

    @classmethod
    def get_model(cls, key: str, loader):
        """Return cached model for `key`. `loader` is a callable to create the model if missing."""
        # Lazy initialize model loaders
        if not cls._loaders_initialized:
            cls._initialize_model_loaders()
            cls._loaders_initialized = True
            
        import time
        current_time = time.time()
        
        if cls._lock:
            cls._lock.acquire()
        try:
            entry = cls._cache.get(key)
            if entry is not None:
                model, ref, _ = entry
                cls._cache[key] = (model, ref + 1, current_time)  # Update last used time
                return model

            # Check if cache is full and evict if needed
            if len(cls._cache) >= cls._max_cache_size:
                # Find least recently used model with ref_count <= 0
                lru_key = None
                lru_time = float('inf')
                for k, (_, ref, last_time) in cls._cache.items():
                    if ref <= 0 and last_time < lru_time:
                        lru_key = k
                        lru_time = last_time
                
                # Evict LRU model if found
                if lru_key:
                    cls._evict_model(lru_key)

            # load model
            model = loader()
            if model is not None:
                cls._cache[key] = (model, 1, current_time)
            return model
        finally:
            if cls._lock:
                cls._lock.release()

    @classmethod
    def _evict_model(cls, key: str):
        """Evict a model from cache with proper cleanup."""
        try:
            entry = cls._cache.get(key)
            if entry:
                model, _, _ = entry
                # Attempt graceful cleanup
                cls._cleanup_model(model)
                cls._cache.pop(key, None)
                # Try to free CUDA cache
                cls._free_cuda_cache()
        except Exception as e:
            from ...utils.monitoring.logger import get_logger
            logger = get_logger(__name__)
            logger.debug(f"Error evicting model {key}: {e}")

    @classmethod
    def _cleanup_model(cls, model):
        """Clean up model resources gracefully."""
        try:
            if hasattr(model, 'to'):
                try:
                    model.to('cpu')
                except Exception:
                    pass
            if hasattr(model, 'eval'):
                try:
                    model.eval()
                except Exception:
                    pass
            if hasattr(model, 'cleanup'):
                try:
                    model.cleanup()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass

    @classmethod
    def _free_cuda_cache(cls):
        """Free CUDA memory cache if available."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @classmethod
    def release_model(cls, key: str):
        """Release a reference to the cached model and free if refcount reaches zero."""
        import time
        current_time = time.time()
        
        if cls._lock:
            cls._lock.acquire()
        try:
            entry = cls._cache.get(key)
            if not entry:
                return
            model, ref, _ = entry
            ref -= 1
            if ref <= 0:
                # Cleanup and remove from cache
                cls._cleanup_model(model)
                cls._cache.pop(key, None)
                cls._free_cuda_cache()
            else:
                cls._cache[key] = (model, ref, current_time)
        finally:
            if cls._lock:
                cls._lock.release()

    @classmethod
    def clear_cache(cls):
        """Clear all cached models."""
        if cls._lock:
            cls._lock.acquire()
        try:
            for key in list(cls._cache.keys()):
                entry = cls._cache.get(key)
                if entry:
                    model, _, _ = entry
                    cls._cleanup_model(model)
            cls._cache.clear()
            cls._free_cuda_cache()
        finally:
            if cls._lock:
                cls._lock.release()

    @classmethod
    def get_cache_status(cls) -> Dict[str, Any]:
        """Get current cache status."""
        if cls._lock:
            cls._lock.acquire()
        try:
            return {
                'current_size': len(cls._cache),
                'max_size': cls._max_cache_size,
                'models': {
                    key: {
                        'ref_count': entry[1],
                        'last_used': entry[2]
                    } 
                    for key, entry in cls._cache.items()
                }
            }
        finally:
            if cls._lock:
                cls._lock.release()

    @classmethod
    def get_from_manager(cls, model_name: str, model_type: str = 'yolo', download: bool = True, verify_hash: bool = True):
        """Get model from ModelManager and cache it in memory.
        
        Args:
            model_name: Name of the model to get
            model_type: Type of model (yolo, clip, detr, pose, etc.)
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
                # Use appropriate loader based on model type
                loader_func = cls._model_loaders.get(model_type.lower())
                if loader_func:
                    try:
                        return loader_func(model_path)
                    except Exception as e:
                        from ...utils.monitoring.logger import get_logger
                        logger = get_logger(__name__)
                        logger.debug(f"Error loading {model_type} model {model_name}: {e}")
            return None
        
        return cls.get_model(f"{model_type}:{model_name}", loader)

    @classmethod
    def register_model_loader(cls, model_type: str, loader_func):
        """Register a custom model loader for a specific model type."""
        if cls._lock:
            cls._lock.acquire()
        try:
            cls._model_loaders[model_type.lower()] = loader_func
        finally:
            if cls._lock:
                cls._lock.release()



