"""
DETR detector implementation
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Union
from .base_detector import BaseDetector
from visionframework.data.detection import Detection
from visionframework.utils.monitoring.logger import get_logger
from visionframework.utils.io.config_models import DeviceManager, ModelCache

logger = get_logger(__name__)

try:
    import torch
    from transformers import DetrImageProcessor, DetrForObjectDetection
    DETR_AVAILABLE = True
except ImportError:
    DETR_AVAILABLE = False
    DetrImageProcessor = None
    DetrForObjectDetection = None


class DETRDetector(BaseDetector):
    """
    DETR detector implementation
    
    This class implements object detection using DETR (Detection Transformer) models
    from HuggingFace Transformers. DETR is an end-to-end object detection model that
    uses transformers instead of anchor-based methods.
    
    Example:
        ```python
        # Using default DETR model
        detector = DETRDetector({
            "model_name": "facebook/detr-resnet-50",
            "conf_threshold": 0.5,
            "device": "cuda"
        })
        detector.initialize()
        detections = detector.detect(image)
        
        # Using larger model
        detector = DETRDetector({
            "model_name": "facebook/detr-resnet-101"
        })
        ```
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize DETR detector
        
        Args:
            config: Configuration dictionary with keys:
                - model_name: DETR model name from HuggingFace (default: 'facebook/detr-resnet-50')
                  Options:
                    - 'facebook/detr-resnet-50': ResNet-50 backbone (faster, smaller)
                    - 'facebook/detr-resnet-101': ResNet-101 backbone (more accurate, larger)
                  Model will be automatically downloaded on first use.
                - conf_threshold: Confidence threshold between 0.0 and 1.0 (default: 0.5)
                  Detections with confidence below this threshold are filtered out.
                - device: Device to use for inference, one of:
                    - 'cpu': CPU inference (default)
                    - 'cuda': GPU inference (requires CUDA-capable GPU)
                    - 'mps': Apple Silicon GPU (macOS only)
        
        Raises:
            ValueError: If configuration is invalid (will be logged as warning)
        """
        super().__init__(config)
        self.model: Optional[Any] = None  # DetrForObjectDetection type from transformers
        self.processor: Optional[Any] = None  # DetrImageProcessor type from transformers
        self.model_name: str = self.config.get("model_name", "facebook/detr-resnet-50")
        self.conf_threshold: float = float(self.config.get("conf_threshold", 0.5))
        self.device: str = self.config.get("device", "cpu")
        perf = self.config.get("performance", {})
        self.batch_inference: bool = bool(self.config.get("batch_inference", perf.get("batch_inference", False)))
        self.use_fp16: bool = bool(self.config.get("use_fp16", perf.get("use_fp16", False)))
        self._cached_model_key: Optional[str] = None
        self._cached_processor_key: Optional[str] = None
    
    def initialize(self) -> bool:
        """Initialize the DETR model"""
        try:
            if not DETR_AVAILABLE:
                raise ImportError("transformers and torch not installed. Install with: pip install transformers torch")
            
            # Use cache for processor and model
            proc_key = f"detr_proc:{self.model_name}"
            mdl_key = f"detr_model:{self.model_name}"
            self.processor = ModelCache.get_model(proc_key, lambda: DetrImageProcessor.from_pretrained(self.model_name))
            self._cached_processor_key = proc_key
            self.model = ModelCache.get_model(mdl_key, lambda: DetrForObjectDetection.from_pretrained(self.model_name))
            self._cached_model_key = mdl_key

            # Normalize device and move model if supported
            device = DeviceManager.normalize_device(self.device)
            if device != self.device:
                logger.info(f"Requested device '{self.device}' not available, using '{device}' instead")
            self.device = device
            try:
                if hasattr(self.model, 'to'):
                    self.model.to(self.device)
            except Exception:
                logger.debug("Model.to(device) not supported or failed; continuing")
            try:
                if hasattr(self.model, 'eval'):
                    self.model.eval()
            except Exception:
                pass
            self.is_initialized = True
            logger.info(f"DETR detector initialized successfully with model: {self.model_name}")
            return True
        except ImportError as e:
            logger.error(f"Missing dependency for DETR detector: {e}", exc_info=True)
            return False
        except (RuntimeError, OSError) as e:
            logger.error(f"Failed to load DETR model ({self.model_name}): {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing DETR detector: {e}", exc_info=True)
            return False

    def cleanup(self) -> None:
        """Release model and processor resources and free GPU memory if possible."""
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

            if self._cached_processor_key:
                try:
                    ModelCache.release_model(self._cached_processor_key)
                except Exception as e:
                    logger.warning(f"Error releasing cached processor '{self._cached_processor_key}': {e}")
                self._cached_processor_key = None
                self.processor = None
            else:
                if self.processor is not None:
                    try:
                        del self.processor
                    except Exception:
                        self.processor = None
                    self.processor = None

            if DETR_AVAILABLE and torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        finally:
            self.is_initialized = False
    
    def detect(self, image: np.ndarray, categories: Optional[Union[list, tuple]] = None) -> List[Detection]:
        """
        Detect objects using DETR
        
        This method performs object detection on the input image using the DETR model.
        DETR uses a transformer architecture and does not require anchor boxes or NMS.
        
        Args:
            image: Input image in BGR format (OpenCV standard), numpy array with shape (H, W, 3).
                   Data type should be uint8 with values in range [0, 255].
        
        Returns:
            List[Detection]: List of Detection objects, each containing:
                - bbox: Tuple of (x1, y1, x2, y2) coordinates
                - confidence: Confidence score between 0.0 and 1.0
                - class_id: Integer class ID (COCO class IDs)
                - class_name: String class name from COCO dataset
                - mask: None (DETR does not support segmentation)
        
        Raises:
            RuntimeError: If model is not initialized or detection fails
            ValueError: If image format is invalid
        
        Note:
            DETR models are pre-trained on COCO dataset and detect 91 classes.
            The model outputs are already filtered by conf_threshold during post-processing.
        
        Example:
            ```python
            detector = DETRDetector()
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
                logger.error("DETR detector not initialized")
                return []
        
        detections: List[Detection] = []
        
        try:
            from PIL import Image

            # Prepare PIL images
            if is_batch:
                pil_images = []
                sizes = []
                for img in image:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(img_rgb)
                    pil_images.append(pil)
                    sizes.append(pil.size[::-1])
                inputs = self.processor(images=pil_images, return_tensors="pt")
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                inputs = self.processor(images=pil_image, return_tensors="pt")

            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Run inference with optional fp16 autocast
            if self.use_fp16 and self.device == 'cuda':
                amp_ctx = torch.cuda.amp.autocast()
            else:
                amp_ctx = torch.no_grad()

            with amp_ctx:
                outputs = self.model(**inputs)

            # Post-process results
            if is_batch:
                target_sizes = torch.tensor(sizes).to(self.device)
                post = self.processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=self.conf_threshold
                )
                # post is list per image
                grouped_results = post
            else:
                target_sizes = torch.tensor([pil_image.size[::-1]]).to(self.device)
                post = self.processor.post_process_object_detection(
                    outputs,
                    target_sizes=target_sizes,
                    threshold=self.conf_threshold
                )
                grouped_results = [post[0]]
            
            # Convert to Detection objects
            for res in grouped_results:
                # Each res corresponds to an image
                for score, label, box in zip(res["scores"], res["labels"], res["boxes"]):
                    # Results are already filtered by threshold, but double-check
                    s_val = float(score.cpu().numpy()) if hasattr(score, 'cpu') else float(score)
                    if s_val >= self.conf_threshold:
                        box_arr = box.cpu().numpy() if hasattr(box, 'cpu') else box
                        cls_id = int(label.cpu().numpy()) if hasattr(label, 'cpu') else int(label)
                        cls_name = self.model.config.id2label[cls_id] if hasattr(self.model.config, 'id2label') else str(cls_id)

                        # Category filtering support
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
                                bbox=(float(box_arr[0]), float(box_arr[1]), float(box_arr[2]), float(box_arr[3])),
                                confidence=s_val,
                                class_id=cls_id,
                                class_name=cls_name
                            )
                            detections.append(detection)

            # If batch input, return list of lists (grouped by image), else flat list
            if is_batch:
                # naive grouping: split detections evenly by number of grouped_results sizes
                grouped: List[List[Detection]] = []
                idx = 0
                for res in grouped_results:
                    cnt = len(res["boxes"]) if res.get("boxes") is not None else 0
                    grouped.append(detections[idx:idx+cnt])
                    idx += cnt
                return grouped
            return detections
            
        except RuntimeError as e:
            logger.error(f"Runtime error during DETR detection: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error during DETR detection: {e}", exc_info=True)
            return []
    
    def detect_batch(self, images: List[np.ndarray], categories: Optional[Union[list, tuple]] = None) -> List[List[Detection]]:
        """
        Detect objects in multiple images using DETR batch processing
        
        This method efficiently processes multiple images in a single batch,
        which is more efficient than processing them individually.
        
        Args:
            images: List of input images in BGR format (numpy arrays, shape: (H, W, 3))
            categories: Optional list of category IDs or names to filter detections
        
        Returns:
            List[List[Detection]]: List of detection lists, one per image.
        
        Example:
            ```python
            detector = DETRDetector()
            detector.initialize()
            images = [frame1, frame2, frame3]
            results = detector.detect_batch(images)
            ```
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("DETR detector not initialized")
                return [[] for _ in images]
        
        if not images:
            logger.warning("Empty image list provided to detect_batch")
            return []
        
        batch_detections: List[List[Detection]] = [[] for _ in images]
        
        try:
            from PIL import Image
            
            # Prepare PIL images and convert
            pil_images = []
            for img in images:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(img_rgb)
                pil_images.append(pil)
            
            # Process batch through model
            inputs = self.processor(images=pil_images, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Run inference
            if self.use_fp16 and self.device == 'cuda':
                amp_ctx = torch.cuda.amp.autocast()
            else:
                amp_ctx = torch.no_grad()
            
            with amp_ctx:
                outputs = self.model(**inputs)
            
            # Process results for each image
            target_sizes = torch.tensor([[img.shape[0], img.shape[1]] for img in images])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self.conf_threshold
            )
            
            # Extract detections for each image
            for img_idx, result in enumerate(results):
                detections: List[Detection] = []
                
                if "scores" in result and len(result["scores"]) > 0:
                    scores = result["scores"].cpu().numpy()
                    labels = result["labels"].cpu().numpy()
                    boxes = result["boxes"].cpu().numpy()
                    
                    for box, label, score in zip(boxes, labels, scores):
                        if score >= self.conf_threshold:
                            cls_id = int(label)
                            cls_name = self.model.config.id2label.get(cls_id, f"class_{cls_id}")
                            
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
                                    confidence=float(score),
                                    class_id=cls_id,
                                    class_name=cls_name
                                )
                                detections.append(detection)
                
                batch_detections[img_idx] = detections
            
            return batch_detections
        
        except Exception as e:
            logger.error(f"Error during DETR batch detection: {e}", exc_info=True)
            return [[] for _ in images]

