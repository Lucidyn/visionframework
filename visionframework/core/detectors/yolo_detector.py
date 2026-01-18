"""
YOLO detector implementation
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Union
from .base_detector import BaseDetector
from ...data.detection import Detection
from ...utils.logger import get_logger

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
                  For segmentation: yolov8n-seg.pt, etc.
                - conf_threshold: Confidence threshold between 0.0 and 1.0 (default: 0.25)
                  Detections with confidence below this threshold are filtered out.
                - iou_threshold: IoU threshold for Non-Maximum Suppression (NMS) (default: 0.45)
                  Must be between 0.0 and 1.0. Higher values allow more overlapping boxes.
                - device: Device to use for inference, one of:
                    - 'cpu': CPU inference (default)
                    - 'cuda': GPU inference (requires CUDA-capable GPU)
                    - 'mps': Apple Silicon GPU (macOS only)
                - enable_segmentation: Enable instance segmentation (default: False)
                  If True and model_path doesn't contain "seg", will try to load segmentation model.
        
        Raises:
            ValueError: If configuration is invalid (will be logged as warning)
        """
        super().__init__(config)
        self.model: Optional[Any] = None  # YOLO model type from ultralytics
        self.model_path: str = self.config.get("model_path", "yolov8n.pt")
        self.conf_threshold: float = float(self.config.get("conf_threshold", 0.25))
        self.iou_threshold: float = float(self.config.get("iou_threshold", 0.45))
        self.device: str = self.config.get("device", "cpu")
        self.enable_segmentation: bool = self.config.get("enable_segmentation", False)
        # Performance options
        perf = self.config.get("performance", {})
        self.batch_inference: bool = bool(self.config.get("batch_inference", perf.get("batch_inference", False)))
        self.use_fp16: bool = bool(self.config.get("use_fp16", perf.get("use_fp16", False)))
    
    def initialize(self) -> bool:
        """Initialize the YOLO model"""
        try:
            if not YOLO_AVAILABLE:
                raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
            
            # Load segmentation model if enabled
            if self.enable_segmentation and "seg" not in self.model_path.lower():
                seg_path = self.model_path.replace(".pt", "-seg.pt")
                try:
                    self.model = YOLO(seg_path)
                    logger.debug(f"Loaded segmentation model: {seg_path}")
                except (FileNotFoundError, RuntimeError) as e:
                    # Fallback to regular model
                    logger.warning(f"Failed to load segmentation model, falling back to regular model: {e}")
                    self.model = YOLO(self.model_path)
            else:
                self.model = YOLO(self.model_path)
            
            self.model.to(self.device)
            self.is_initialized = True
            logger.info(f"YOLO detector initialized successfully with model: {self.model_path}")
            return True
        except ImportError as e:
            logger.error(f"Missing dependency for YOLO detector: {e}", exc_info=True)
            return False
        except (FileNotFoundError, RuntimeError) as e:
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
        
        detections: List[Detection] = []
        
        try:
            # Run YOLO inference (supports single image or batch)
            is_batch = isinstance(image, (list, tuple)) or (isinstance(image, np.ndarray) and image.ndim == 4)
            results = None
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
                    results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
            else:
                results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
            
            # Process results
            # results is iterable over images
            for img_idx, result in enumerate(results):
                boxes = result.boxes
                masks = result.masks if hasattr(result, 'masks') and result.masks is not None else None

                if boxes is not None:
                    for i in range(len(boxes)):
                        # Extract bounding box
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        cls_name = result.names[cls_id] if hasattr(result, 'names') else str(cls_id)

                        # Get segmentation mask if available
                        mask = None
                        if masks is not None and i < len(masks.data):
                            mask_data = masks.data[i].cpu().numpy()
                            # determine source image for mask sizing
                            src_img = image[img_idx] if is_batch and isinstance(image, (list, tuple)) else (image if not is_batch else None)
                            if src_img is not None:
                                h, w = src_img.shape[:2]
                                if mask_data.shape != (h, w):
                                    mask_data = cv2.resize(mask_data.astype(np.float32), (w, h))
                                mask = (mask_data > 0.5).astype(np.uint8) * 255

                        # Create Detection object
                            # Category filtering support: accept int ids or string names
                            keep = True
                            if categories is not None:
                                keep = False
                                for c in categories:
                                    if isinstance(c, (int,)) and c == cls_id:
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
                                    class_name=cls_name,
                                    mask=mask
                                )
                                detections.append(detection)

            # If batch inference, return list-of-lists: group detections per image
            if is_batch:
                # naive grouping by iteration: ultralytics returns results per image in same order
                grouped: List[List[Detection]] = []
                idx = 0
                for result in results:
                    count = len(result.boxes) if result.boxes is not None else 0
                    grouped.append(detections[idx:idx+count])
                    idx += count
                return grouped
            
            return detections
            
        except RuntimeError as e:
            logger.error(f"Runtime error during YOLO detection: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error during YOLO detection: {e}", exc_info=True)
            return []

