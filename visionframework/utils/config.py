"""
Configuration utilities

This module provides configuration loading and management utilities,
supporting both JSON and YAML formats.
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path

# Try to import yaml, but make it optional
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None


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
        return {
            "model_path": "yolov8n.pt",
            "model_type": "yolo",
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "device": "cpu"
        }
    
    @staticmethod
    def get_default_tracker_config() -> Dict[str, Any]:
        """Get default tracker configuration"""
        return {
            "max_age": 30,
            "min_hits": 3,
            "iou_threshold": 0.3,
            "use_kalman": False
        }
    
    @staticmethod
    def get_default_pipeline_config() -> Dict[str, Any]:
        """Get default pipeline configuration"""
        return {
            "enable_tracking": True,
            "detector_config": Config.get_default_detector_config(),
            "tracker_config": Config.get_default_tracker_config()
        }
    
    @staticmethod
    def get_default_visualizer_config() -> Dict[str, Any]:
        """Get default visualizer configuration"""
        return {
            "show_labels": True,
            "show_confidences": True,
            "show_track_ids": True,
            "line_thickness": 2,
            "font_scale": 0.5
        }

