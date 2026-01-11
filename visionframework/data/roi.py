"""
ROI data structure
"""

import cv2
import numpy as np
from typing import List, Tuple


class ROI:
    """Region of Interest container"""
    
    def __init__(
        self,
        name: str,
        points: List[Tuple[float, float]],
        roi_type: str = "polygon"
    ):
        """
        Initialize ROI
        
        Args:
            name: ROI name/ID
            points: List of (x, y) points defining the ROI
            roi_type: Type of ROI ('polygon', 'rectangle', 'circle')
        """
        self.name = name
        self.points = points
        self.type = roi_type
        self.mask = None
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside ROI"""
        if self.type == "polygon":
            return cv2.pointPolygonTest(
                np.array(self.points, dtype=np.float32),
                point,
                False
            ) >= 0
        elif self.type == "rectangle":
            if len(self.points) != 2:
                return False
            x1, y1 = self.points[0]
            x2, y2 = self.points[1]
            x, y = point
            return x1 <= x <= x2 and y1 <= y <= y2
        elif self.type == "circle":
            if len(self.points) != 2:
                return False
            center, radius_point = self.points[0], self.points[1]
            cx, cy = center
            rx, ry = radius_point
            radius = np.sqrt((rx - cx)**2 + (ry - cy)**2)
            x, y = point
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            return dist <= radius
        return False
    
    def contains_bbox(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if bounding box center is inside ROI"""
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        return self.contains_point(center)
    
    def get_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Get binary mask for ROI"""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        if self.type == "polygon":
            pts = np.array(self.points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        elif self.type == "rectangle":
            x1, y1 = map(int, self.points[0])
            x2, y2 = map(int, self.points[1])
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        elif self.type == "circle":
            center = tuple(map(int, self.points[0]))
            radius_point = self.points[1]
            cx, cy = center
            rx, ry = radius_point
            radius = int(np.sqrt((rx - cx)**2 + (ry - cy)**2))
            cv2.circle(mask, center, radius, 255, -1)
        
        self.mask = mask
        return mask

