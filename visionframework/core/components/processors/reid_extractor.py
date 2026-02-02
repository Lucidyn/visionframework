"""
ReID Feature Extractor

Provides person re-identification feature extraction.
"""

from typing import List, Optional, Any, Tuple, Union
import numpy as np
import cv2

try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import resnet50, ResNet50_Weights
except ImportError:
    torch = None

from .feature_extractor import FeatureExtractor
from visionframework.utils.monitoring.logger import get_logger
from visionframework.utils.io.config_models import Config, ModelCache

logger = get_logger(__name__)


class ReIDExtractor(FeatureExtractor):
    """
    Person Re-Identification Feature Extractor
    
    Extracts appearance features (embeddings) from image crops using ResNet50
    or custom ReID models. Can be used for person tracking and re-identification.
    """
    
    def __init__(self, model_name: str = "resnet50", device: str = "cpu",
                 input_size: Tuple[int, int] = (128, 256),
                 model_path: Optional[str] = None,
                 use_pretrained: bool = True,
                 use_fp16: bool = False):
        """
        Initialize ReID extractor.
        
        Args:
            model_name: Model architecture name
            device: Device to run on ("cpu", "cuda", etc.)
            input_size: Input size as (width, height)
            model_path: Path to custom model weights
            use_pretrained: Whether to use pretrained weights
            use_fp16: Whether to use FP16 precision for inference
        """
        super().__init__(model_name, device)
        self.input_size = input_size
        self.model_path = model_path
        self.use_pretrained = use_pretrained
        self.use_fp16 = use_fp16
        self._cached_model_key: Optional[str] = None
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(input_size[::-1]),  # (H, W) for Resize
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
    
    def initialize(self) -> None:
        """Initialize the ReID model."""
        if torch is None:
            raise ImportError("PyTorch is required for ReID extraction. "
                            "Install with: pip install 'visionframework[reid]'")
        
        try:
            # Build cache key based on model_name, pretrained flag, model_path and use_fp16
            key = f"reid:{self.model_name}:pretrained={self.use_pretrained}:path={self.model_path or 'none'}:fp16={self.use_fp16}"

            def loader():
                # loader should create model on CPU to allow moving to target device per-extractor
                from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, \
                                            ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, \
                                            ResNet101_Weights, ResNet152_Weights
                
                # Map model name to architecture and weights
                model_map = {
                    'resnet18': (resnet18, ResNet18_Weights.IMAGENET1K_V1 if self.use_pretrained else None),
                    'resnet34': (resnet34, ResNet34_Weights.IMAGENET1K_V1 if self.use_pretrained else None),
                    'resnet50': (resnet50, ResNet50_Weights.IMAGENET1K_V1 if self.use_pretrained else None),
                    'resnet101': (resnet101, ResNet101_Weights.IMAGENET1K_V1 if self.use_pretrained else None),
                    'resnet152': (resnet152, ResNet152_Weights.IMAGENET1K_V1 if self.use_pretrained else None),
                }
                
                # Get model architecture and weights
                model_class, weights = model_map.get(self.model_name, (resnet50, ResNet50_Weights.IMAGENET1K_V1 if self.use_pretrained else None))
                
                if self.model_path:
                    logger.info(f"Loading custom ReID model from {self.model_path}")
                    base_model = model_class(weights=None)
                    state_dict = torch.load(self.model_path, map_location='cpu')
                    base_model.load_state_dict(state_dict)
                else:
                    logger.info(f"Loading ReID feature extractor ({self.model_name})")
                    base_model = model_class(weights=weights)

                feat_model = torch.nn.Sequential(*list(base_model.children())[:-1])
                feat_model.to('cpu')
                feat_model.eval()
                return feat_model

            # Obtain (or load) cached feature model
            self.model = ModelCache.get_model(key, loader)
            self._cached_model_key = key

            # Move to desired device for this extractor instance
            try:
                if hasattr(self.model, 'to'):
                    self.model.to(self.device)
                    # Enable FP16 if requested and supported
                    if self.use_fp16 and torch.cuda.is_available():
                        self.model = self.model.half()
            except Exception:
                logger.debug("Failed to move ReID model to device; continuing")

            self._initialized = True
            logger.info("ReID extractor initialized")
        except Exception as e:
            logger.error(f"Failed to initialize ReID extractor: {e}", exc_info=True)
            raise RuntimeError(f"ReID initialization failed: {e}")
    
    def extract(self, image: np.ndarray, 
                bboxes: Optional[List[Tuple[float, float, float, float]]] = None) -> np.ndarray:
        """
        Extract features for bounding boxes.
        
        Args:
            image: Full frame image (BGR)
            bboxes: List of (x1, y1, x2, y2) bounding boxes
        
        Returns:
            Feature embeddings shape (N, feature_dim)
        """
        return self.process(image, bboxes or [])
    
    def process(self, image: np.ndarray, 
                bboxes: List[Tuple[float, float, float, float]]) -> np.ndarray:
        """
        Extract features for multiple bounding boxes.
        
        Args:
            image: Full frame image (BGR)
            bboxes: List of (x1, y1, x2, y2) bounding boxes
        
        Returns:
            Feature embeddings shape (N, 2048)
        """
        if not self.is_initialized():
            self.initialize()
        
        if not bboxes:
            return np.empty((0, 2048))
        
        # Crop images from bboxes
        crops = []
        h, w = image.shape[:2]
        
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                # Invalid crop
                crop = np.zeros((self.input_size[1], self.input_size[0], 3), 
                               dtype=np.uint8)
            else:
                crop = image[y1:y2, x1:x2]
                # Convert BGR to RGB
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            crops.append(self.transform(crop))
        
        if not crops:
            return np.empty((0, 2048))
        
        # Forward pass
        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            features = self.model(batch)
            # Flatten (N, 2048, 1, 1) -> (N, 2048)
            features = features.view(features.size(0), -1)
            # L2 normalize
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        return features.cpu().numpy()
    
    def process_batch(self, images: List[np.ndarray], 
                     bboxes_list: List[List[Tuple[float, float, float, float]]]) -> List[np.ndarray]:
        """
        Extract features for multiple images with bounding boxes (batch processing).
        
        This method efficiently processes multiple images at once, which is more efficient
        than processing them individually, especially when the number of bounding boxes varies.
        
        Args:
            images: List of full frame images (BGR)
            bboxes_list: List of bounding box lists, one list per image.
                        Each list contains (x1, y1, x2, y2) tuples.
        
        Returns:
            List[np.ndarray]: List of feature matrices, one per image.
                            Each matrix has shape (N, 2048) where N is the number of boxes in that image.
        
        Example:
            ```python
            reid = ReIDExtractor()
            reid.initialize()
            
            images = [frame1, frame2, frame3]
            bboxes_list = [
                [(10, 20, 100, 200), (150, 50, 250, 300)],  # 2 boxes in frame1
                [(5, 10, 95, 180)],                          # 1 box in frame2
                [(20, 30, 120, 210), (200, 100, 300, 400)]  # 2 boxes in frame3
            ]
            features = reid.process_batch(images, bboxes_list)
            # features[0].shape = (2, 2048)
            # features[1].shape = (1, 2048)
            # features[2].shape = (2, 2048)
            ```
        """
        if not self.is_initialized():
            self.initialize()
        
        batch_features = []
        
        # Process each image individually
        for image, bboxes in zip(images, bboxes_list):
            if not bboxes:
                batch_features.append(np.empty((0, 2048)))
            else:
                features = self.process(image, bboxes)
                batch_features.append(features)
        
        return batch_features
    
    def _move_to_device(self, device: str) -> None:
        """Move model to device."""
        if self.model is not None:
            try:
                self.model.to(device)
            except Exception:
                logger.debug("ReID model move to device failed")

    def cleanup(self) -> None:
        """Release or release-reference cached model."""
        try:
            if self._cached_model_key:
                try:
                    ModelCache.release_model(self._cached_model_key)
                except Exception as e:
                    logger.warning(f"Error releasing cached ReID model '{self._cached_model_key}': {e}")
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
        finally:
            self._initialized = False
