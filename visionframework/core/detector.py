"""
Unified detector interface

This module provides a unified interface for different object detection models,
including YOLO, DETR, and RF-DETR. Users can switch between models by simply
changing the model_type configuration parameter.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .base import BaseModule
from ..data.detection import Detection
from ..utils.logger import get_logger
from .detectors.yolo_detector import YOLODetector
from .detectors.detr_detector import DETRDetector
from .detectors.rfdetr_detector import RFDETRDetector

logger = get_logger(__name__)


class Detector(BaseModule):
    """
    Unified detector interface supporting multiple backends
    
    This class provides a unified interface for object detection using different
    models (YOLO, DETR, RF-DETR). It automatically handles model-specific initialization
    and provides a consistent API regardless of the underlying model.
    
    Example:
        ```python
        # Using YOLO
        detector = Detector({
            "model_type": "yolo",
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.25
        })
        detector.initialize()
        detections = detector.detect(image)
        
        # Using DETR
        detector = Detector({
            "model_type": "detr",
            "detr_model_name": "facebook/detr-resnet-50",
            "conf_threshold": 0.5
        })
        
        # Using RF-DETR
        detector = Detector({
            "model_type": "rfdetr",
            "conf_threshold": 0.5
        })
        ```
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize detector
        
        Args:
            config: Configuration dictionary with keys:
                - model_path: Path to model file (default: 'yolov8n.pt')
                  Only used for YOLO models. First use will download automatically.
                - model_type: Type of model, one of:
                    - 'yolo': YOLOv8 model (default)
                    - 'detr': DETR (Detection Transformer) model
                    - 'rfdetr': RF-DETR (Roboflow DETR) model
                - conf_threshold: Confidence threshold between 0.0 and 1.0 (default: 0.25)
                - iou_threshold: IoU threshold for NMS, between 0.0 and 1.0 (default: 0.45)
                  Only used for YOLO models.
                - device: Device to use, one of 'cpu', 'cuda', 'mps' (default: 'cpu')
                - detr_model_name: DETR model name for HuggingFace (default: 'facebook/detr-resnet-50')
                  Options: 'facebook/detr-resnet-50', 'facebook/detr-resnet-101'
                - rfdetr_model_name: RF-DETR model name (default: None, uses default model)
                - enable_segmentation: Enable segmentation for YOLO (default: False)
        
        Raises:
            ValueError: If configuration is invalid (will be logged as warning)
        """
        super().__init__(config)
        self.detector_impl: Optional[BaseModule] = None
        self.model_type: str = self.config.get("model_type", "yolo")
        self.model_path: str = self.config.get("model_path", "yolov8n.pt")
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate detector configuration
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            Tuple[bool, Optional[str]]: 
                - (True, None) if valid
                - (False, error_message) if invalid
        
        Validates:
            - model_type is one of the supported types
            - conf_threshold is between 0.0 and 1.0
            - iou_threshold is between 0.0 and 1.0 (if provided)
            - device is a valid device type
        """
        # Validate model_type
        if "model_type" in config:
            model_type = config["model_type"]
            if model_type not in ["yolo", "detr", "rfdetr"]:
                return False, f"Invalid model_type: {model_type}. Supported: 'yolo', 'detr', 'rfdetr'"
        
        # Validate conf_threshold
        if "conf_threshold" in config:
            conf_threshold = config["conf_threshold"]
            if not isinstance(conf_threshold, (int, float)):
                return False, f"conf_threshold must be a number, got {type(conf_threshold).__name__}"
            if not 0.0 <= conf_threshold <= 1.0:
                return False, f"conf_threshold must be between 0.0 and 1.0, got {conf_threshold}"
        
        # Validate iou_threshold
        if "iou_threshold" in config:
            iou_threshold = config["iou_threshold"]
            if not isinstance(iou_threshold, (int, float)):
                return False, f"iou_threshold must be a number, got {type(iou_threshold).__name__}"
            if not 0.0 <= iou_threshold <= 1.0:
                return False, f"iou_threshold must be between 0.0 and 1.0, got {iou_threshold}"
        
        # Validate device
        if "device" in config:
            device = config["device"]
            if device not in ["cpu", "cuda", "mps"]:
                return False, f"Invalid device: {device}. Supported: 'cpu', 'cuda', 'mps'"
        
        return True, None
    
    def initialize(self) -> bool:
        """
        Initialize the detector model
        
        This method initializes the underlying detector implementation based on
        the configured model_type. It performs model loading, device configuration,
        and any necessary setup steps.
        
        Returns:
            bool: True if initialization successful, False otherwise.
                  On failure, errors are logged with detailed information.
        
        Raises:
            ValueError: If model_type is invalid or configuration is incorrect
            ImportError: If required dependencies are missing (e.g., ultralytics, transformers)
            RuntimeError: If model loading or initialization fails
        
        Note:
            Initialization may involve:
            - Downloading model files (first use)
            - Loading model weights into memory
            - Moving model to specified device (CPU/GPU)
            - Setting model to evaluation mode
        
        Example:
            ```python
            detector = Detector({"model_type": "yolo", "model_path": "yolov8n.pt"})
            if detector.initialize():
                print("Detector initialized successfully")
            else:
                print("Initialization failed, check logs for details")
            ```
        """
        try:
            if self.model_type == "yolo":
                self.detector_impl = YOLODetector(self.config)
            elif self.model_type == "detr":
                detr_config = self.config.copy()
                detr_config["model_name"] = self.config.get("detr_model_name", "facebook/detr-resnet-50")
                self.detector_impl = DETRDetector(detr_config)
            elif self.model_type == "rfdetr":
                rfdetr_config = self.config.copy()
                rfdetr_config["model_name"] = self.config.get("rfdetr_model_name", None)
                self.detector_impl = RFDETRDetector(rfdetr_config)
            else:
                raise ValueError(f"Unsupported model_type: {self.model_type}. Supported: 'yolo', 'detr', 'rfdetr'")
            
            return self.detector_impl.initialize()
        except ValueError as e:
            logger.error(f"Invalid detector configuration: {e}", exc_info=True)
            return False
        except (ImportError, RuntimeError) as e:
            logger.error(f"Failed to initialize detector ({self.model_type}): {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing detector: {e}", exc_info=True)
            return False
    
    def process(self, image: np.ndarray, categories: Optional[list] = None) -> List[Detection]:
        """
        Detect objects in image
        
        This method processes an input image and returns a list of detected objects.
        If the detector is not initialized, it will attempt to initialize automatically.
        
        Args:
            image: Input image in BGR format (OpenCV standard), numpy array with shape (H, W, 3)
                   Data type should be uint8 with values in range [0, 255]
        
        Returns:
            List[Detection]: List of Detection objects, each containing:
                - bbox: Tuple of (x1, y1, x2, y2) coordinates
                - confidence: Confidence score between 0.0 and 1.0
                - class_id: Integer class ID
                - class_name: String class name
                - mask: Optional segmentation mask (if segmentation is enabled)
            
            Returns empty list if:
                - Detector is not initialized and initialization fails
                - detector_impl is None
                - No objects are detected
                - An error occurs during detection
        
        Raises:
            RuntimeError: If detector is not initialized and automatic initialization fails
            ValueError: If image is invalid (wrong format, shape, or data type)
        
        Example:
            ```python
            detector = Detector()
            detector.initialize()
            detections = detector.process(image)
            for det in detections:
                print(f"{det.class_name}: {det.confidence:.2f}")
            ```
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("Detector not initialized and auto-initialization failed")
                return []
        
        if self.detector_impl is None:
            logger.warning("detector_impl is None, returning empty list")
            return []
        
        try:
            return self.detector_impl.detect(image, categories=categories)
        except Exception as e:
            logger.error(f"Error during detection: {e}", exc_info=True)
            return []
    
    def detect(self, image: np.ndarray, categories: Optional[list] = None) -> List[Detection]:
        """
        Alias for process method
        
        This method is provided for convenience and clarity. It is functionally
        equivalent to process().
        
        Args:
            image: Input image in BGR format (numpy array, shape: (H, W, 3))
        
        Returns:
            List[Detection]: List of detected objects
        """
        return self.process(image, categories=categories)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model
        
        Returns information about the currently configured detector, including
        model type, path, device, and thresholds. If the detector is not initialized,
        returns a status indicator.
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - status: "initialized" or "not_initialized"
                - model_type: Type of model ('yolo', 'detr', 'rfdetr')
                - model_path: Path to model file (for YOLO)
                - device: Device being used ('cpu', 'cuda', 'mps')
                - conf_threshold: Confidence threshold
                - iou_threshold: IoU threshold (for YOLO)
            
            If not initialized, returns: {"status": "not_initialized"}
        
        Example:
            ```python
            detector = Detector({"model_type": "yolo"})
            info = detector.get_model_info()
            print(f"Model type: {info.get('model_type')}")
            
            detector.initialize()
            info = detector.get_model_info()
            print(f"Device: {info.get('device')}")
            ```
        """
        if not self.is_initialized:
            return {"status": "not_initialized"}
        
        info: Dict[str, Any] = {
            "status": "initialized",
            "model_type": self.model_type,
            "model_path": self.model_path,
            "device": self.config.get("device", "cpu"),
            "conf_threshold": self.config.get("conf_threshold", 0.25),
            "iou_threshold": self.config.get("iou_threshold", 0.45),
        }
        
        # Add model-specific information
        if self.model_type == "detr":
            info["detr_model_name"] = self.config.get("detr_model_name", "facebook/detr-resnet-50")
        elif self.model_type == "rfdetr":
            info["rfdetr_model_name"] = self.config.get("rfdetr_model_name", None)
        
        return info
