"""
ReID feature extractor module
"""

import cv2
import numpy as np
from typing import List, Optional, Union, Dict, Any, Tuple
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from .base import BaseModule
from ..utils.logger import get_logger

logger = get_logger(__name__)


class ReIDExtractor(BaseModule):
    """
    Re-Identification Feature Extractor
    
    Extracts appearance features (embeddings) from image crops using a deep learning model.
    Defaults to a ResNet50 model pre-trained on ImageNet if no specific ReID model is provided.
    For better performance, use a model trained on person ReID datasets (like Market-1501).
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ReID extractor
        
        Args:
            config: Configuration dictionary with keys:
                - model_path: Path to custom model weights (optional)
                - device: 'cpu' or 'cuda' (default: 'cpu')
                - input_size: Tuple (width, height) for model input (default: (128, 256))
        """
        super().__init__(config)
        self.device = self.config.get("device", "cpu")
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
            
        self.input_size = self.config.get("input_size", (128, 256))
        self.model_path = self.config.get("model_path", None)
        self.use_pretrained = self.config.get("use_pretrained", True)
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize(self.input_size[::-1]), # (H, W) for Resize, but input_size is (W, H)
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = None
        self.is_initialized = False

    def initialize(self) -> bool:
        """Initialize the model"""
        try:
            if self.model_path:
                logger.info(f"Loading custom ReID model from {self.model_path}...")
                # Load custom model logic here if needed
                # For now we still use ResNet50 structure
                base_model = resnet50(weights=None)
                state_dict = torch.load(self.model_path, map_location=self.device)
                base_model.load_state_dict(state_dict)
            else:
                logger.info("Loading ReID feature extractor (ResNet50)...")
                weights = ResNet50_Weights.IMAGENET1K_V1 if self.use_pretrained else None
                base_model = resnet50(weights=weights)
                
            self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
            
            self.model.to(self.device)
            self.model.eval()
            self.is_initialized = True
            logger.info("ReID extractor initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ReID extractor: {e}", exc_info=True)
            return False

    def process(self, image: np.ndarray, bboxes: List[Tuple[float, float, float, float]]) -> np.ndarray:
        """
        Extract features for multiple bounding boxes
        
        Args:
            image: Full frame image (BGR)
            bboxes: List of (x1, y1, x2, y2) tuples
            
        Returns:
            np.ndarray: Feature matrix of shape (N, 2048) where N is number of boxes
        """
        if not self.is_initialized:
            if not self.initialize():
                return np.empty((0, 2048))
                
        if not bboxes:
            return np.empty((0, 2048))
            
        crops = []
        h, w = image.shape[:2]
        
        for box in bboxes:
            x1, y1, x2, y2 = map(int, box)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                # Invalid crop, add black image
                crop = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
            else:
                crop = image[y1:y2, x1:x2]
                # Convert BGR to RGB
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                
            crops.append(self.transform(crop))
            
        if not crops:
            return np.empty((0, 2048))
            
        batch = torch.stack(crops).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
            # ResNet50 output is (N, 2048, 1, 1), flatten to (N, 2048)
            features = features.view(features.size(0), -1)
            # Normalize features
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
        return features.cpu().numpy()

    def extract(self, image: np.ndarray, bboxes: List[Tuple[float, float, float, float]]) -> np.ndarray:
        """Alias for process"""
        return self.process(image, bboxes)
