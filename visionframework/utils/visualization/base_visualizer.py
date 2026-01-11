"""
Base visualizer class
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional


class BaseVisualizer:
    """Base class for visualizers"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize visualizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.line_thickness = self.config.get("line_thickness", 2)
        self.font_scale = self.config.get("font_scale", 0.5)
        self.color_palette = self._generate_color_palette()
    
    def _generate_color_palette(self, n_colors: int = 100) -> List[Tuple[int, int, int]]:
        """Generate a color palette"""
        colors = []
        for i in range(n_colors):
            hue = int(180 * i / n_colors)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for a class ID"""
        return self.color_palette[class_id % len(self.color_palette)]

