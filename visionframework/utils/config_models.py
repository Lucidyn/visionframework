"""
Pydantic Configuration Models

This module defines Pydantic models for configuration validation, providing
stronger type checking and validation than traditional dictionaries.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, field_validator


class BaseConfig(BaseModel):
    """Base configuration model with common validation"""
    class Config:
        extra = "allow"  # Allow additional fields not explicitly defined
        from_attributes = True  # Support loading from objects


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


def validate_config(config_dict: Dict[str, Any], model_type: BaseConfig) -> BaseConfig:
    """
    Validate configuration dictionary using Pydantic model
    
    Args:
        config_dict: Configuration dictionary to validate
        model_type: Pydantic model class to use for validation
        
    Returns:
        BaseConfig: Validated configuration model instance
        
    Raises:
        pydantic.ValidationError: If validation fails
    """
    return model_type(**config_dict)
