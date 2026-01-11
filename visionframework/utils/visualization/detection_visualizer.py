"""
Detection visualization
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from .base_visualizer import BaseVisualizer
from ...data.detection import Detection


class DetectionVisualizer(BaseVisualizer):
    """Visualizer for detection results"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.show_labels = self.config.get("show_labels", True)
        self.show_confidences = self.config.get("show_confidences", True)
    
    def draw_detection(
        self,
        image: np.ndarray,
        detection: Detection,
        color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """Draw a single detection on image"""
        x1, y1, x2, y2 = map(int, detection.bbox)
        
        if color is None:
            color = self._get_color(detection.class_id)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.line_thickness)
        
        # Draw mask if available
        if detection.has_mask():
            mask = detection.mask
            color_mask = np.zeros_like(image)
            color_mask[mask > 0] = color
            image = cv2.addWeighted(image, 1.0, color_mask, 0.3, 0)
        
        # Prepare label
        label_parts = []
        if self.show_labels and detection.class_name:
            label_parts.append(detection.class_name)
        if self.show_confidences:
            label_parts.append(f"{detection.confidence:.2f}")
        
        if label_parts:
            label = " ".join(label_parts)
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.line_thickness
            )
            
            # Draw label background
            cv2.rectangle(
                image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                self.line_thickness
            )
        
        return image
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection]
    ) -> np.ndarray:
        """Draw multiple detections on image"""
        result = image.copy()
        for detection in detections:
            result = self.draw_detection(result, detection)
        return result

