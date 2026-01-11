"""
Detection data structure
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


class Detection:
    """Detection result container"""
    
    def __init__(
        self,
        bbox: Tuple[float, float, float, float],  # (x1, y1, x2, y2)
        confidence: float,
        class_id: int,
        class_name: Optional[str] = None,
        mask: Optional[np.ndarray] = None  # Segmentation mask
    ):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.mask = mask  # Binary mask for instance segmentation
    
    def __repr__(self):
        return f"Detection(class={self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name
        }
        if self.mask is not None:
            result["has_mask"] = True
        return result
    
    def has_mask(self) -> bool:
        """Check if detection has segmentation mask"""
        return self.mask is not None

