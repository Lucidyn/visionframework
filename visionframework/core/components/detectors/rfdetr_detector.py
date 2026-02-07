"""
RF-DETR detector implementation
"""

import cv2
import numpy as np
from contextlib import nullcontext
from typing import List, Optional, Dict, Any, Union
from .base_detector import BaseDetector
from visionframework.data.detection import Detection
from visionframework.utils.monitoring.logger import get_logger
from visionframework.utils.io.config_models import DeviceManager, ModelCache

logger = get_logger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from rfdetr import RFDETRBase
    RFDETR_AVAILABLE = True
except ImportError:
    RFDETR_AVAILABLE = False
    RFDETRBase = None


class RFDETRDetector(BaseDetector):
    """
    RF-DETR detector implementation
    
    This class implements object detection using RF-DETR (Roboflow Detection Transformer)
    models. RF-DETR is a high-performance real-time object detection model developed by
    Roboflow, achieving over 60 AP on COCO dataset and 6ms inference time on edge devices.
    
    Example:
        ```python
        # Using default RF-DETR model
        detector = RFDETRDetector({
            "conf_threshold": 0.5,
            "device": "cuda"
        })
        detector.initialize()
        detections = detector.detect(image)
        
        # Using custom model name (if available)
        detector = RFDETRDetector({
            "model_name": "custom-model-name"
        })
        ```
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize RF-DETR detector
        
        Args:
            config: Configuration dictionary with keys:
                - model_name: RF-DETR model name (default: None, uses default model)
                  If None, the default RF-DETR model will be used.
                  Custom model names can be provided if you have trained custom models.
                - conf_threshold: Confidence threshold between 0.0 and 1.0 (default: 0.5)
                  Detections with confidence below this threshold are filtered out.
                - device: Device to use for inference, one of:
                    - 'cpu': CPU inference (default)
                    - 'cuda': GPU inference (if available)
                    - 'mps': Apple Silicon GPU (macOS only, if available)
                  
                  Note: RF-DETR handles device selection internally based on availability.
        
        Raises:
            ValueError: If configuration is invalid (will be logged as warning)
        """
        super().__init__(config)
        self.model: Optional[Any] = None  # RFDETRBase type from rfdetr
        self.model_name: Optional[str] = self.config.get("model_name", None)
        self.conf_threshold: float = float(self.config.get("conf_threshold", 0.5))
        self.device: str = self.config.get("device", "cpu")
        self._cached_model_key: Optional[str] = None
    
    def initialize(self) -> bool:
        """Initialize the RF-DETR model"""
        try:
            if not RFDETR_AVAILABLE:
                raise ImportError("rfdetr not installed. Install with: pip install rfdetr")
            
            # Initialize RF-DETR model using cache
            key = f"rfdetr:{self.model_name or 'default'}"
            def loader():
                if self.model_name:
                    return RFDETRBase(model_name=self.model_name)
                return RFDETRBase()

            self.model = ModelCache.get_model(key, loader)
            self._cached_model_key = key

            # Move to device if supported
            device = DeviceManager.normalize_device(self.device)
            if device != self.device:
                logger.info(f"Requested device '{self.device}' not available, using '{device}' instead")
            self.device = device
            if hasattr(self.model, 'to'):
                try:
                    self.model.to(self.device)
                except Exception:
                    pass
            
            self.is_initialized = True
            logger.info(f"RF-DETR detector initialized successfully")
            return True
        except ImportError as e:
            logger.error(f"Missing dependency for RF-DETR detector: {e}", exc_info=True)
            return False
        except RuntimeError as e:
            logger.error(f"Failed to initialize RF-DETR model: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing RF-DETR detector: {e}", exc_info=True)
            return False

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
            if TORCH_AVAILABLE and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        finally:
            self.is_initialized = False
    
    def detect(self, image: np.ndarray, categories: Optional[Union[list, tuple]] = None) -> List[Detection]:
        """
        Detect objects using RF-DETR
        
        This method performs object detection on the input image using the RF-DETR model.
        RF-DETR uses supervision library's Detections format internally, which is then
        converted to the framework's Detection objects.
        
        Args:
            image: Input image in BGR format (OpenCV standard), numpy array with shape (H, W, 3).
                   Data type should be uint8 with values in range [0, 255].
        
        Returns:
            List[Detection]: List of Detection objects, each containing:
                - bbox: Tuple of (x1, y1, x2, y2) coordinates
                - confidence: Confidence score between 0.0 and 1.0
                - class_id: Integer class ID
                - class_name: String class name
                - mask: None (RF-DETR does not support segmentation in current implementation)
        
        Raises:
            RuntimeError: If model is not initialized or detection fails
            ValueError: If image format is invalid
        
        Note:
            RF-DETR models are optimized for real-time performance and edge devices.
            The model outputs are filtered by conf_threshold during inference.
        
        Example:
            ```python
            detector = RFDETRDetector()
            detector.initialize()
            detections = detector.detect(image)
            for det in detections:
                print(f"{det.class_name}: {det.confidence:.2f}")
            ```
        """
        # Validate input (skip for batch — validated per-element inside)
        is_batch = isinstance(image, (list, tuple)) or (isinstance(image, np.ndarray) and image.ndim == 4)
        if not is_batch:
            self._validate_image(image)

        if not self.is_initialized:
            if not self.initialize():
                logger.error("RF-DETR detector not initialized")
                return []
        
        detections: List[Detection] = []

        try:
            from PIL import Image

            images = []
            if is_batch:
                for img in image:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(Image.fromarray(img_rgb))
            else:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(img_rgb))

            # Run inference for each image (model.predict may not support batch)
            amp_ctx = None
            if TORCH_AVAILABLE:
                try:
                    if self.config.get('use_fp16', False) and self.device == 'cuda':
                        amp_ctx = torch.cuda.amp.autocast()
                    else:
                        amp_ctx = torch.no_grad()
                except Exception:
                    amp_ctx = None

            results_list = []
            if amp_ctx is not None:
                with amp_ctx:
                    for pil_image in images:
                        res = self.model.predict(pil_image, threshold=self.conf_threshold)
                        results_list.append(res)
            else:
                for pil_image in images:
                    res = self.model.predict(pil_image, threshold=self.conf_threshold)
                    results_list.append(res)

            # Convert supervision-like detections to our Detection objects
            for supervision_detections in results_list:
                # Accept any object with attribute 'xyxy'
                if hasattr(supervision_detections, 'xyxy') and len(supervision_detections.xyxy) > 0:
                    boxes = supervision_detections.xyxy

                    # Extract confidences
                    if hasattr(supervision_detections, 'confidence') and supervision_detections.confidence is not None:
                        confidences = supervision_detections.confidence
                        if hasattr(confidences, '__len__') and not isinstance(confidences, str):
                            confidences = [float(c) for c in confidences]
                        else:
                            confidences = [float(confidences)] * len(boxes)
                    else:
                        confidences = [1.0] * len(boxes)

                    # Extract class IDs
                    if hasattr(supervision_detections, 'class_id') and supervision_detections.class_id is not None:
                        class_ids = supervision_detections.class_id
                        if hasattr(class_ids, '__len__') and not isinstance(class_ids, str):
                            class_ids = [int(cid) for cid in class_ids]
                        else:
                            class_ids = [int(class_ids)] * len(boxes)
                    else:
                        class_ids = [0] * len(boxes)

                    # Extract class names
                    class_names_list = None
                    if hasattr(supervision_detections, 'data') and supervision_detections.data:
                        if isinstance(supervision_detections.data, dict):
                            class_names_list = supervision_detections.data.get('class_name', None)
                        elif hasattr(supervision_detections.data, 'class_name'):
                            class_names_list = supervision_detections.data.class_name

                    # Create Detection objects
                    for i in range(len(boxes)):
                        box = boxes[i]
                        conf = confidences[i] if i < len(confidences) else 1.0
                        cls_id = class_ids[i] if i < len(class_ids) else 0

                        # Get class name
                        cls_name: str
                        if class_names_list and i < len(class_names_list):
                            cls_name = str(class_names_list[i])
                        elif hasattr(self.model, 'class_names') and isinstance(self.model.class_names, (list, tuple)) and cls_id < len(self.model.class_names):
                            cls_name = str(self.model.class_names[cls_id])
                        else:
                            cls_name = f"class_{cls_id}"

                        # Category filtering: support int ids or string names
                        keep = True
                        if categories is not None:
                            keep = False
                            for c in categories:
                                if isinstance(c, int) and c == cls_id:
                                    keep = True
                                    break
                                if isinstance(c, str) and c == cls_name:
                                    keep = True
                                    break

                        if keep:
                            detection = Detection(
                                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                                confidence=conf,
                                class_id=cls_id,
                                class_name=cls_name
                            )
                            detections.append(detection)

            return detections

        except RuntimeError as e:
            logger.error(f"Runtime error during RF-DETR detection: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error during RF-DETR detection: {e}", exc_info=True)
            return []
    
    def detect_batch(self, images: List[np.ndarray], categories: Optional[Union[list, tuple]] = None) -> List[List[Detection]]:
        """
        Detect objects in multiple images using RF-DETR batch processing
        
        This method efficiently processes multiple images in a single batch,
        which is more efficient than processing them individually.
        
        Args:
            images: List of input images in BGR format (numpy arrays, shape: (H, W, 3))
            categories: Optional list of category IDs or names to filter detections
        
        Returns:
            List[List[Detection]]: List of detection lists, one per image.
        
        Example:
            ```python
            detector = RFDETRDetector()
            detector.initialize()
            images = [frame1, frame2, frame3]
            results = detector.detect_batch(images)
            ```
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("RF-DETR detector not initialized")
                return [[] for _ in images]
        
        if not images:
            logger.warning("Empty image list provided to detect_batch")
            return []
        
        batch_detections: List[List[Detection]] = [[] for _ in images]
        
        try:
            import supervision
            
            # Process each image individually (RF-DETR batch inference support varies)
            for img_idx, image in enumerate(images):
                detections: List[Detection] = []
                
                try:
                    # Run inference on single image
                    with torch.no_grad() if TORCH_AVAILABLE else nullcontext():
                        results = self.model(image)
                    
                    # Convert to supervision Detections if needed
                    if hasattr(results, '__class__') and 'Detections' in str(results.__class__):
                        supervision_detections = results
                    else:
                        supervision_detections = supervision.Detections.from_ultralytics(results)
                    
                    # Extract and process detections
                    if hasattr(supervision_detections, 'xyxy') and len(supervision_detections.xyxy) > 0:
                        boxes = supervision_detections.xyxy
                        
                        # Extract confidences
                        if hasattr(supervision_detections, 'confidence') and supervision_detections.confidence is not None:
                            confidences = supervision_detections.confidence
                            if hasattr(confidences, '__len__') and not isinstance(confidences, str):
                                confidences = [float(c) for c in confidences]
                            else:
                                confidences = [float(confidences)] * len(boxes)
                        else:
                            confidences = [1.0] * len(boxes)
                        
                        # Extract class IDs
                        if hasattr(supervision_detections, 'class_id') and supervision_detections.class_id is not None:
                            class_ids = supervision_detections.class_id
                            if hasattr(class_ids, '__len__') and not isinstance(class_ids, str):
                                class_ids = [int(cid) for cid in class_ids]
                            else:
                                class_ids = [int(class_ids)] * len(boxes)
                        else:
                            class_ids = [0] * len(boxes)
                        
                        # Process each detection
                        for box, conf, cls_id in zip(boxes, confidences, class_ids):
                            if conf >= self.conf_threshold:
                                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                                
                                # Category filtering
                                keep = True
                                if categories is not None:
                                    keep = False
                                    for c in categories:
                                        if isinstance(c, int) and c == cls_id:
                                            keep = True
                                            break
                                        if isinstance(c, str) and c == cls_name:
                                            keep = True
                                            break
                                
                                if keep:
                                    detection = Detection(
                                        bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                                        confidence=conf,
                                        class_id=cls_id,
                                        class_name=cls_name
                                    )
                                    detections.append(detection)
                
                except Exception as e:
                    logger.warning(f"Error processing image {img_idx} in batch: {e}")
                
                batch_detections[img_idx] = detections
            
            return batch_detections
        
        except Exception as e:
            logger.error(f"Error during RF-DETR batch detection: {e}", exc_info=True)
            return [[] for _ in images]

