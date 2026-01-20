"""
Image utility functions
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path
from .logger import get_logger

logger = get_logger(__name__)


class ImageUtils:
    """Image utility functions"""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """Load image from file"""
        if not Path(image_path).exists():
            logger.warning(f"Image not found: {image_path}")
            return None
        
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to load image: {image_path}")
            return None
        
        return image
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> bool:
        """Save image to file"""
        try:
            cv2.imwrite(output_path, image)
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False
    
    @staticmethod
    def resize_image(
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        max_size: Optional[int] = None,
        keep_aspect: bool = True
    ) -> np.ndarray:
        """Resize image"""
        h, w = image.shape[:2]
        
        if target_size:
            if keep_aspect:
                scale = min(target_size[0] / w, target_size[1] / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
            else:
                new_w, new_h = target_size
        elif max_size:
            if w > h:
                scale = max_size / w
            else:
                scale = max_size / h
            new_w = int(w * scale)
            new_h = int(h * scale)
        else:
            return image
        
        return cv2.resize(image, (new_w, new_h))
    
    @staticmethod
    def crop_image(
        image: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """Crop image using bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def draw_bbox(
        image: np.ndarray,
        bbox: Tuple[float, float, float, float],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw bounding box on image"""
        x1, y1, x2, y2 = map(int, bbox)
        return cv2.rectangle(image.copy(), (x1, y1), (x2, y2), color, thickness)

