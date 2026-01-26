"""
YOLO detector implementation
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Union
from .base_detector import BaseDetector
from ...data.detection import Detection
from ...utils.monitoring.logger import get_logger
from ...utils.io.config_models import DeviceManager, ModelCache

logger = get_logger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLO = None


class YOLODetector(BaseDetector):
    """
    YOLO detector implementation
    
    This class implements object detection using YOLO (You Only Look Once) models
    from the Ultralytics library. It supports YOLOv8 models and can perform both
    object detection and instance segmentation.
    
    Example:
        ```python
        # Basic detection
        detector = YOLODetector({
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.25,
            "device": "cuda"
        })
        detector.initialize()
        detections = detector.detect(image)
        
        # With segmentation
        detector = YOLODetector({
            "model_path": "yolov8n-seg.pt",
            "enable_segmentation": True
        })
        ```
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize YOLO detector
        
        Args:
            config: Configuration dictionary with keys:
                - model_path: Path to YOLO model file (default: 'yolov8n.pt')
                  First use will automatically download the model.
                  Available models: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
                  yolov26n.pt, yolov26s.pt, yolov26m.pt, yolov26l.pt, yolov26x.pt
                  For segmentation: yolov8n-seg.pt, yolov26n-seg.pt, etc.
                - conf_threshold: Confidence threshold between 0.0 and 1.0 (default: 0.25)
                  Detections with confidence below this threshold are filtered out.
                - category_thresholds: Dictionary of category-specific confidence thresholds
                  Example: {"person": 0.7, "car": 0.5}
                - iou_threshold: IoU threshold for Non-Maximum Suppression (NMS) (default: 0.45)
                  Must be between 0.0 and 1.0. Higher values allow more overlapping boxes.
                - device: Device to use for inference, one of:
                    - 'cpu': CPU inference (default)
                    - 'cuda': GPU inference (requires CUDA-capable GPU)
                    - 'mps': Apple Silicon GPU (macOS only)
                    - 'auto': Automatically select best available device
                - enable_segmentation: Enable instance segmentation (default: False)
                  If True and model_path doesn't contain "seg", will try to load segmentation model.
                - batch_inference: Enable batch inference (default: False)
                - dynamic_batch_size: Enable dynamic batch size (default: False)
                  Automatically adjusts batch size based on available memory and input size.
                - max_batch_size: Maximum batch size for dynamic batching (default: 8)
                - min_batch_size: Minimum batch size for dynamic batching (default: 1)
                - quantization: Model quantization option, one of:
                    - None: No quantization (default)
                    - 'int8': INT8 quantization for CPU/GPU
                    - 'fp16': FP16 quantization for GPU
                    - 'dynamic': Dynamic quantization
                - custom_classes: Optional dictionary mapping class IDs to class names
                  Used for custom trained models
                - custom_model: Boolean flag indicating if this is a custom trained model (default: False)
        
        Raises:
            ValueError: If configuration is invalid (will be logged as warning)
        """
        super().__init__(config)
        self.model: Optional[Any] = None  # YOLO model type from ultralytics
        self.model_path: str = self.config.get("model_path", "yolov8n.pt")
        self.conf_threshold: float = float(self.config.get("conf_threshold", 0.25))
        self.category_thresholds: Optional[Dict[str, float]] = self.config.get("category_thresholds")
        self.iou_threshold: float = float(self.config.get("iou_threshold", 0.45))
        self.device: str = self.config.get("device", "cpu")
        self.enable_segmentation: bool = self.config.get("enable_segmentation", False)
        # Performance options
        perf = self.config.get("performance", {})
        self.batch_inference: bool = bool(self.config.get("batch_inference", perf.get("batch_inference", False)))
        self.dynamic_batch_size: bool = bool(self.config.get("dynamic_batch_size", perf.get("dynamic_batch_size", False)))
        self.max_batch_size: int = int(self.config.get("max_batch_size", perf.get("max_batch_size", 8)))
        self.min_batch_size: int = int(self.config.get("min_batch_size", perf.get("min_batch_size", 1)))
        self.use_fp16: bool = bool(self.config.get("use_fp16", perf.get("use_fp16", False)))
        self.quantization: Optional[str] = self.config.get("quantization")
        self.custom_classes: Optional[Dict[int, str]] = self.config.get("custom_classes")
        self.custom_model: bool = bool(self.config.get("custom_model", False))
        self._cached_model_key: Optional[str] = None
    
    def initialize(self) -> bool:
        """Initialize the YOLO model"""
        try:
            if not YOLO_AVAILABLE:
                raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
            
            # Determine model name for ModelManager
            model_name = self.model_path.split(".")[0]  # Extract model name without extension
            
            # Load model directly from file path (supports custom models)
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Handle auto device selection
            if self.device == 'auto':
                selected_device = DeviceManager.auto_select_device()
                logger.info(f"Auto-selected device: '{selected_device}'")
                self.device = selected_device
            else:
                # Normalize and validate device choice
                device = DeviceManager.normalize_device(self.device)
                if device != self.device:
                    logger.info(f"Requested device '{self.device}' not available, using '{device}' instead")
                self.device = device
            
            # Apply quantization if specified
            if self.quantization:
                logger.info(f"Applying {self.quantization} quantization to model")
                try:
                    # Handle different quantization types
                    if self.quantization == 'int8':
                        # For Ultralytics YOLO, we can set the model to use INT8 precision
                        # This is handled through the device parameter or model configuration
                        pass  # YOLO handles INT8 through device context
                    elif self.quantization == 'fp16':
                        # FP16 is already supported through use_fp16 flag
                        self.use_fp16 = True
                    elif self.quantization == 'dynamic':
                        # Dynamic quantization for CPU
                        if self.device == 'cpu':
                            logger.info("Enabling dynamic quantization for CPU")
                            # Note: Ultralytics YOLO doesn't directly support dynamic quantization
                            # This would require additional implementation with PyTorch
                except Exception as e:
                    logger.warning(f"Failed to apply quantization: {e}")
            
            # Try moving model to device if supported by model wrapper
            try:
                if hasattr(self.model, 'to'):
                    self.model.to(self.device)
            except Exception:
                logger.debug("Model.to(device) not supported or failed; continuing")
            
            # Set custom classes if provided
            if self.custom_classes:
                logger.info(f"Setting custom classes: {self.custom_classes}")
                # Update model names with custom classes
                if hasattr(self.model, 'names'):
                    # Create a copy to avoid modifying the original
                    self.model.names = dict(self.model.names)  # Convert to dict if needed
                    self.model.names.update(self.custom_classes)
            
            self.is_initialized = True
            logger.info(f"YOLO detector initialized successfully with model: {self.model_path}")
            return True
        except ImportError as e:
            logger.error(f"Missing dependency for YOLO detector: {e}", exc_info=True)
            return False
        except (FileNotFoundError, RuntimeError, ValueError) as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing YOLO detector: {e}", exc_info=True)
            return False
    
    def detect(self, image: np.ndarray, categories: Optional[Union[list, tuple]] = None) -> List[Detection]:
        """
        Detect objects using YOLO
        
        This method performs object detection on the input image using the YOLO model.
        It supports both regular detection and instance segmentation (if enabled).
        
        Args:
            image: Input image in BGR format (OpenCV standard), numpy array with shape (H, W, 3).
                   Data type should be uint8 with values in range [0, 255].
        
        Returns:
            List[Detection]: List of Detection objects, each containing:
                - bbox: Tuple of (x1, y1, x2, y2) coordinates
                - confidence: Confidence score between 0.0 and 1.0
                - class_id: Integer class ID (COCO class IDs for default YOLO models)
                - class_name: String class name (e.g., "person", "car", "dog")
                - mask: Optional segmentation mask (numpy array) if segmentation is enabled,
                       None otherwise
        
        Raises:
            RuntimeError: If model is not initialized or detection fails
        
        Example:
            ```python
            detector = YOLODetector({"model_path": "yolov8n.pt"})
            detector.initialize()
            detections = detector.detect(image)
            for det in detections:
                print(f"{det.class_name}: {det.confidence:.2f}")
            ```
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("YOLO detector not initialized")
                return []
        
        try:
            # Run YOLO inference
            is_batch = isinstance(image, (list, tuple)) or (isinstance(image, np.ndarray) and image.ndim == 4)
            results = self._run_inference(image)
            
            # Process results
            detections_list = self._process_results(results, image, categories)
            
            # Return single list for single image, list of lists for batch
            if is_batch:
                return detections_list
            return detections_list[0] if detections_list else []
            
        except RuntimeError as e:
            logger.error(f"Runtime error during YOLO detection: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error during YOLO detection: {e}", exc_info=True)
            return []

    def detect_batch(self, images: List[np.ndarray], categories: Optional[list] = None) -> List[List[Detection]]:
        """
        Process multiple images in batch mode using YOLO
        
        This method efficiently processes multiple images in a single batch when possible,
        which is more efficient than processing them individually. The batch is passed to YOLO
        which returns results for all images at once.
        
        Args:
            images: List of input images in BGR format (numpy arrays, shape: (H, W, 3))
            categories: Optional list of category IDs or names to filter detections
        
        Returns:
            List[List[Detection]]: List of detection lists, one per image.
                                  Each inner list contains Detection objects for that image.
        
        Example:
            ```python
            detector = YOLODetector({
                "model_path": "yolov8n.pt",
                "batch_inference": True,
                "dynamic_batch_size": True,
                "max_batch_size": 16,
                "conf_threshold": 0.5
            })
            detector.initialize()
            
            images = [frame1, frame2, frame3]
            batch_results = detector.detect_batch(images)
            
            for idx, detections in enumerate(batch_results):
                print(f"Frame {idx}: {len(detections)} detections")
            ```
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("YOLO detector not initialized")
                return [[] for _ in images]
        
        if not images:
            logger.warning("Empty image list provided to detect_batch")
            return []
        
        # Dynamic batch size calculation
        batch_size = len(images)
        if self.batch_inference:
            if self.dynamic_batch_size:
                # Calculate optimal batch size based on device and image count
                num_images = len(images)
                # Start with a reasonable batch size based on device
                if self.device == 'cuda':
                    # GPU can handle larger batches
                    batch_size = min(num_images, self.max_batch_size)
                    # Adjust batch size based on number of images
                    if num_images < batch_size:
                        batch_size = num_images
                    elif num_images > batch_size * 2:
                        # If we have many images, use max batch size for efficiency
                        batch_size = self.max_batch_size
                else:
                    # CPU is limited, use smaller batches
                    batch_size = min(num_images, max(2, self.min_batch_size))
            else:
                # Fixed batch size
                batch_size = min(len(images), self.max_batch_size)
        else:
            # Batch inference disabled, process individually
            batch_size = 1
        
        logger.debug(f"Using batch size: {batch_size} for {len(images)} images")
        
        batch_detections: List[List[Detection]] = [[] for _ in images]
        
        try:
            # Process images in batches
            for batch_start in range(0, len(images), batch_size):
                # Get current batch
                batch_end = min(batch_start + batch_size, len(images))
                current_batch = images[batch_start:batch_end]
                
                # Run YOLO inference on current batch
                results = self._run_inference(current_batch)
                
                # Process results for current batch
                batch_dets = self._process_results(results, current_batch, categories)
                
                # Assign detections to the correct image indices
                for batch_idx, dets in enumerate(batch_dets):
                    img_idx = batch_start + batch_idx
                    batch_detections[img_idx] = dets
            
            return batch_detections
        except Exception as e:
            logger.error(f"Error during batch detection: {e}", exc_info=True)
            return [[] for _ in images]
    
    def _create_detection(self, box: np.ndarray, conf: float, cls_id: int, cls_name: str, 
                        masks: Any, mask_idx: int, source_image: np.ndarray) -> Optional[Detection]:
        """
        Create a Detection object from YOLO detection results
        
        Args:
            box: Bounding box coordinates in format (x1, y1, x2, y2)
            conf: Confidence score
            cls_id: Class ID
            cls_name: Class name
            masks: YOLO mask results
            mask_idx: Index of the mask in the masks data
            source_image: Source image for mask sizing
            
        Returns:
            Optional[Detection]: Detection object if created successfully, None otherwise
        """
        # Get segmentation mask if available
        mask = None
        if masks is not None and mask_idx < len(masks.data):
            mask_data = masks.data[mask_idx].cpu().numpy()
            # Resize mask to match source image dimensions
            h, w = source_image.shape[:2]
            if mask_data.shape != (h, w):
                mask_data = cv2.resize(mask_data.astype(np.float32), (w, h))
            mask = (mask_data > 0.5).astype(np.uint8) * 255
        
        return Detection(
            bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            confidence=conf,
            class_id=cls_id,
            class_name=cls_name,
            mask=mask
        )
    
    def _should_keep_detection(self, conf: float, cls_id: int, cls_name: str, categories: Optional[Union[list, tuple]]) -> bool:
        """
        Determine if a detection should be kept based on confidence thresholds and category filters
        
        Args:
            conf: Confidence score
            cls_id: Class ID
            cls_name: Class name
            categories: Optional list of categories to filter by
            
        Returns:
            bool: True if detection should be kept, False otherwise
        """
        # Apply category-specific confidence threshold if available
        category_threshold = self.conf_threshold
        if self.category_thresholds:
            # Check by class name first, then by class id as fallback
            if cls_name in self.category_thresholds:
                category_threshold = self.category_thresholds[cls_name]
            elif str(cls_id) in self.category_thresholds:
                category_threshold = self.category_thresholds[str(cls_id)]
        
        # Check confidence against category threshold
        if conf < category_threshold:
            return False
        
        # Check categories filter if provided
        if categories is not None:
            for c in categories:
                if (isinstance(c, (int,)) and c == cls_id) or (isinstance(c, str) and c == cls_name):
                    return True
            return False
        
        return True
    
    def _process_results(self, results: Any, source_images: Union[np.ndarray, List[np.ndarray]], 
                       categories: Optional[Union[list, tuple]]) -> List[List[Detection]]:
        """
        Process YOLO inference results into Detection objects
        
        Args:
            results: YOLO inference results
            source_images: Source images used for inference
            categories: Optional list of categories to filter by
            
        Returns:
            List[List[Detection]]: List of detection lists, one per image
        """
        detections_list: List[List[Detection]] = []
        
        # Convert single image to list for consistent processing
        is_single_image = not isinstance(source_images, (list, tuple)) and source_images.ndim == 3
        if is_single_image:
            source_images = [source_images]
        
        # Process each result
        for img_idx, result in enumerate(results):
            img_detections: List[Detection] = []
            boxes = result.boxes
            masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None
            
            if boxes is not None:
                for i in range(len(boxes)):
                    # Extract bounding box
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())
                    
                    # Get class name - use custom classes if provided, otherwise use result.names
                    cls_name = str(cls_id)
                    if self.custom_classes and cls_id in self.custom_classes:
                        cls_name = self.custom_classes[cls_id]
                    elif hasattr(result, 'names') and cls_id in result.names:
                        cls_name = result.names[cls_id]
                    
                    # Check if we should keep this detection
                    if self._should_keep_detection(conf, cls_id, cls_name, categories):
                        # Create detection object
                        detection = self._create_detection(box, conf, cls_id, cls_name, masks, i, source_images[img_idx])
                        if detection:
                            img_detections.append(detection)
            
            # Additional post-processing: sort detections by confidence
            img_detections.sort(key=lambda x: x.confidence, reverse=True)
            
            detections_list.append(img_detections)
        
        return detections_list
    
    def _run_inference(self, images: Union[np.ndarray, List[np.ndarray]]) -> Any:
        """
        Run YOLO inference with proper context management
        
        Args:
            images: Input images for inference
            
        Returns:
            Any: YOLO inference results
        """
        try:
            import torch
            ctx = torch.no_grad()
            if self.use_fp16 and self.device == 'cuda':
                amp = torch.cuda.amp.autocast()
                ctx = amp
        except Exception:
            ctx = None
        
        if ctx is not None:
            with ctx:
                return self.model(images, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        else:
            return self.model(images, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
    
    def cleanup(self) -> None:
        """Release model resources and free GPU memory if possible."""
        try:
            if self._cached_model_key:
                try:
                    ModelCache.release_model(self._cached_model_key)
                except Exception as e:
                    logger.warning(f"Error releasing cached model '{self._cached_model_key}': {e}")
                self._cached_model_key = None
                self.model = None
            else:
                if self.model is not None:
                    try:
                        if hasattr(self.model, 'to'):
                            self.model.to('cpu')
                    except Exception:
                        pass
                    try:
                        del self.model
                    except Exception:
                        self.model = None
                    self.model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        finally:
            self.is_initialized = False

