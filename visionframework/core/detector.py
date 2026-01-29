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
from ..utils.monitoring.logger import get_logger
from .segmenters import SAMSegmenter
from ..utils import error_handler

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
                - category_thresholds: Dictionary of category-specific confidence thresholds
                  Example: {"person": 0.7, "car": 0.5}
                - iou_threshold: IoU threshold for NMS, between 0.0 and 1.0 (default: 0.45)
                  Only used for YOLO models.
                - device: Device to use, one of 'cpu', 'cuda', 'mps', 'auto' (default: 'cpu')
                - detr_model_name: DETR model name for HuggingFace (default: 'facebook/detr-resnet-50')
                  Options: 'facebook/detr-resnet-50', 'facebook/detr-resnet-101'
                - rfdetr_model_name: RF-DETR model name (default: None, uses default model)
                - enable_segmentation: Enable segmentation for YOLO (default: False)
                - quantization: Model quantization option, one of: None, 'int8', 'fp16', 'dynamic'
                - custom_classes: Optional dictionary mapping class IDs to class names
                  Used for custom trained models
                - custom_model: Boolean flag indicating if this is a custom trained model (default: False)
                - segmenter_type: Type of segmenter, one of: None, 'sam' (default: None)
                - sam_model_path: Path to SAM model file (default: None, downloads automatically)
                - sam_model_type: SAM model type, one of: 'vit_h', 'vit_l', 'vit_b' (default: 'vit_b')
                - sam_use_fp16: Whether to use FP16 precision for SAM (default: True for CUDA, False for CPU)
        
        Raises:
            ValueError: If configuration is invalid (will be logged as warning)
        """
        super().__init__(config)
        self.detector_impl: Optional[BaseModule] = None
        self.model_type: str = self.config.get("model_type", "yolo")
        self.model_path: str = self.config.get("model_path", "yolov8n.pt")
        
        # Segmenter integration
        self.segmenter: Optional[SAMSegmenter] = None
        self.segmenter_type: Optional[str] = self.config.get("segmenter_type")
        self.sam_model_path: Optional[str] = self.config.get("sam_model_path")
        self.sam_model_type: str = self.config.get("sam_model_type", "vit_b")
        self.sam_use_fp16: bool = self.config.get("sam_use_fp16", self.config.get("device") == "cuda")
    
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
            - category_thresholds is a dictionary (if provided)
            - quantization is a valid option (if provided)
            - custom_classes is a dictionary (if provided)
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
            if device not in ["cpu", "cuda", "mps", "auto"]:
                return False, f"Invalid device: {device}. Supported: 'cpu', 'cuda', 'mps', 'auto'"
        
        # Validate category_thresholds
        if "category_thresholds" in config:
            category_thresholds = config["category_thresholds"]
            if not isinstance(category_thresholds, dict):
                return False, f"category_thresholds must be a dictionary, got {type(category_thresholds).__name__}"
            # Validate each threshold in category_thresholds
            for category, threshold in category_thresholds.items():
                if not isinstance(threshold, (int, float)):
                    return False, f"Threshold for category '{category}' must be a number, got {type(threshold).__name__}"
                if not 0.0 <= threshold <= 1.0:
                    return False, f"Threshold for category '{category}' must be between 0.0 and 1.0, got {threshold}"
        
        # Validate quantization
        if "quantization" in config:
            quantization = config["quantization"]
            if quantization not in [None, "int8", "fp16", "dynamic"]:
                return False, f"Invalid quantization option: {quantization}. Supported: None, 'int8', 'fp16', 'dynamic'"
        
        # Validate custom_classes
        if "custom_classes" in config:
            custom_classes = config["custom_classes"]
            if not isinstance(custom_classes, dict):
                return False, f"custom_classes must be a dictionary, got {type(custom_classes).__name__}"
            # Validate each entry in custom_classes
            for class_id, class_name in custom_classes.items():
                if not isinstance(class_id, int):
                    return False, f"Class ID in custom_classes must be an integer, got {type(class_id).__name__}"
                if not isinstance(class_name, str):
                    return False, f"Class name in custom_classes must be a string, got {type(class_name).__name__}"
        
        # Validate custom_model
        if "custom_model" in config:
            custom_model = config["custom_model"]
            if not isinstance(custom_model, bool):
                return False, f"custom_model must be a boolean, got {type(custom_model).__name__}"
        
        # Validate segmenter_type
        if "segmenter_type" in config:
            segmenter_type = config["segmenter_type"]
            if segmenter_type not in [None, "sam"]:
                return False, f"Invalid segmenter_type: {segmenter_type}. Supported: None, 'sam'"
        
        # Validate sam_model_type
        if "sam_model_type" in config:
            sam_model_type = config["sam_model_type"]
            if sam_model_type not in ["vit_h", "vit_l", "vit_b"]:
                return False, f"Invalid sam_model_type: {sam_model_type}. Supported: 'vit_h', 'vit_l', 'vit_b'"
        
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
            # Lazy import detector implementations to avoid importing heavy libraries at module load time
            if self.model_type == "yolo":
                from .detectors.yolo_detector import YOLODetector
                self.detector_impl = YOLODetector(self.config)
            elif self.model_type == "detr":
                from .detectors.detr_detector import DETRDetector
                detr_config = self.config.copy()
                detr_config["model_name"] = self.config.get("detr_model_name", "facebook/detr-resnet-50")
                self.detector_impl = DETRDetector(detr_config)
            elif self.model_type == "rfdetr":
                from .detectors.rfdetr_detector import RFDETRDetector
                rfdetr_config = self.config.copy()
                rfdetr_config["model_name"] = self.config.get("rfdetr_model_name", None)
                self.detector_impl = RFDETRDetector(rfdetr_config)
            else:
                from ..exceptions import ConfigurationError
                error_handler.handle_error(
                    ValueError(f"Unsupported model_type: {self.model_type}"),
                    ConfigurationError,
                    f"Unsupported model type",
                    {"model_type": self.model_type, "supported_types": ["yolo", "detr", "rfdetr"]}
                )
                return False
            
            result = self.detector_impl.initialize()
            if result:
                # Initialize segmenter if configured
                if self.segmenter_type == "sam":
                    self.segmenter = SAMSegmenter({
                        "model_type": self.sam_model_type,
                        "model_path": self.sam_model_path,
                        "device": self.config.get("device", "cpu"),
                        "use_fp16": self.sam_use_fp16
                    })
                    segmenter_result = self.segmenter.initialize()
                    if not segmenter_result:
                        logger.warning("Failed to initialize SAM segmenter, continuing with detection only")
                        self.segmenter = None
                self.is_initialized = True
            return result
        except ValueError as e:
            from ..exceptions import ConfigurationError
            error_handler.handle_error(
                e,
                ConfigurationError,
                "Invalid detector configuration",
                {"model_type": self.model_type, "model_path": self.model_path}
            )
            return False
        except ImportError as e:
            from ..exceptions import DependencyError
            error_handler.handle_error(
                e,
                DependencyError,
                "Failed to import detector dependencies",
                {"model_type": self.model_type}
            )
            return False
        except RuntimeError as e:
            from ..exceptions import DetectorInitializationError
            error_handler.handle_error(
                e,
                DetectorInitializationError,
                f"Failed to initialize detector",
                {"model_type": self.model_type, "model_path": self.model_path}
            )
            return False
        except Exception as e:
            from ..exceptions import DetectorInitializationError
            error_handler.handle_error(
                e,
                DetectorInitializationError,
                "Unexpected error initializing detector",
                {"model_type": self.model_type, "model_path": self.model_path}
            )
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
        """
        if not self.is_initialized:
            if not self.initialize():
                from ..exceptions import DetectorInitializationError
                error_handler.handle_error(
                    RuntimeError("Auto-initialization failed"),
                    DetectorInitializationError,
                    "Detector initialization failed",
                    {"model_type": self.model_type, "model_path": self.model_path}
                )
                return []
        
        if self.detector_impl is None:
            from ..exceptions import DetectorInitializationError
            error_handler.handle_error(
                ValueError("Detector implementation is None"),
                DetectorInitializationError,
                "Detector implementation not initialized",
                {"model_type": self.model_type}
            )
            return []
        
        try:
            # Validate input image
            is_valid, error_msg = error_handler.validate_input(
                image,
                np.ndarray,
                "image"
            )
            if not is_valid:
                from ..exceptions import DataFormatError
                error_handler.handle_error(
                    ValueError(error_msg),
                    DataFormatError,
                    "Invalid input image",
                    {"error": error_msg}
                )
                return []
            
            detections = self.detector_impl.detect(image, categories=categories)
            
            # If segmenter is available, perform segmentation on detections
            if self.segmenter is not None and detections:
                detections = self.segmenter.segment_detections(image, detections)
            
            return detections
        except Exception as e:
            from ..exceptions import DetectorInferenceError
            error_handler.handle_error(
                e,
                DetectorInferenceError,
                "Error during detection",
                {"model_type": self.model_type, "input_shape": image.shape if hasattr(image, 'shape') else None}
            )
            return []
    
    def process_batch(self, images: List[np.ndarray], categories: Optional[list] = None) -> List[List[Detection]]:
        """
        Process multiple images in batch mode
        
        This method processes multiple images efficiently by utilizing batch inference
        when the underlying detector supports it (e.g., YOLO with batch_inference enabled).
        For detectors without batch support, images are processed sequentially.
        
        Args:
            images: List of input images in BGR format (numpy arrays, shape: (H, W, 3))
            categories: Optional list of category names for filtering detections
        
        Returns:
            List[List[Detection]]: List of detection lists, one list per image.
                                  Each inner list contains Detection objects for that image.
        """
        if not self.is_initialized:
            if not self.initialize():
                from ..exceptions import DetectorInitializationError
                error_handler.handle_error(
                    RuntimeError("Auto-initialization failed"),
                    DetectorInitializationError,
                    "Detector initialization failed",
                    {"model_type": self.model_type, "model_path": self.model_path}
                )
                return [[] for _ in images]
        
        if self.detector_impl is None:
            from ..exceptions import DetectorInitializationError
            error_handler.handle_error(
                ValueError("Detector implementation is None"),
                DetectorInitializationError,
                "Detector implementation not initialized",
                {"model_type": self.model_type}
            )
            return [[] for _ in images]
        
        try:
            # Check if detector has batch detect method
            if hasattr(self.detector_impl, 'detect_batch'):
                all_detections = self.detector_impl.detect_batch(images, categories=categories)
            else:
                # Fallback: process images individually
                logger.debug("Detector does not support batch detection, processing individually")
                all_detections = []
                for image in images:
                    dets = self.detector_impl.detect(image, categories=categories)
                    all_detections.append(dets)
            
            # If segmenter is available, perform segmentation on detections
            if self.segmenter is not None:
                for i, detections in enumerate(all_detections):
                    if detections:
                        all_detections[i] = self.segmenter.segment_detections(images[i], detections)
            
            return all_detections
        except Exception as e:
            from ..exceptions import DetectorInferenceError
            error_handler.handle_error(
                e,
                DetectorInferenceError,
                "Error during batch detection",
                {"model_type": self.model_type, "batch_size": len(images)}
            )
            return [[] for _ in images]
    
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
                - quantization: Quantization setting (if any)
                - custom_model: Whether this is a custom trained model
                - custom_classes_count: Number of custom classes (if any)
                - enable_segmentation: Whether segmentation is enabled
            
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
            "quantization": self.config.get("quantization", None),
            "custom_model": self.config.get("custom_model", False),
            "enable_segmentation": self.config.get("enable_segmentation", False),
        }
        
        # Add custom classes count if available
        custom_classes = self.config.get("custom_classes", {})
        if custom_classes:
            info["custom_classes_count"] = len(custom_classes)
        
        # Add model-specific information
        if self.model_type == "detr":
            info["detr_model_name"] = self.config.get("detr_model_name", "facebook/detr-resnet-50")
        elif self.model_type == "rfdetr":
            info["rfdetr_model_name"] = self.config.get("rfdetr_model_name", None)
        
        return info

    def cleanup(self) -> None:
        """Cleanup resources held by detector and underlying implementation."""
        try:
            # Cleanup segmenter first if available
            if self.segmenter:
                try:
                    self.segmenter.cleanup()
                except Exception as e:
                    logger.warning(f"Error during segmenter.cleanup(): {e}")
                self.segmenter = None
            
            # Cleanup detector implementation
            if self.detector_impl and hasattr(self.detector_impl, 'cleanup'):
                try:
                    self.detector_impl.cleanup()
                except Exception as e:
                    logger.warning(f"Error during detector_impl.cleanup(): {e}")
        finally:
            self.detector_impl = None
            self.segmenter = None
            self.is_initialized = False
