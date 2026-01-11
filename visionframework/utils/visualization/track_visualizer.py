"""
Track visualization
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from .base_visualizer import BaseVisualizer
from ...data.track import Track


class TrackVisualizer(BaseVisualizer):
    """Visualizer for track results"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.show_labels = self.config.get("show_labels", True)
        self.show_confidences = self.config.get("show_confidences", True)
        self.show_track_ids = self.config.get("show_track_ids", True)
    
    def draw_track(
        self,
        image: np.ndarray,
        track: Track,
        color: Optional[Tuple[int, int, int]] = None,
        draw_history: bool = True
    ) -> np.ndarray:
        """Draw a single track on image"""
        x1, y1, x2, y2 = map(int, track.bbox)
        
        if color is None:
            color = self._get_color(track.class_id)
        
        # Draw track history
        if draw_history and len(track.history) > 1:
            points = []
            for hist_bbox in track.history[-10:]:  # Last 10 positions
                cx = int((hist_bbox[0] + hist_bbox[2]) / 2)
                cy = int((hist_bbox[1] + hist_bbox[3]) / 2)
                points.append((cx, cy))
            
            if len(points) > 1:
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    line_color = tuple(int(c * alpha) for c in color)
                    cv2.line(image, points[i-1], points[i], line_color, 2)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.line_thickness)
        
        # Prepare label
        label_parts = []
        if self.show_track_ids:
            label_parts.append(f"ID:{track.track_id}")
        if self.show_labels and track.class_name:
            label_parts.append(track.class_name)
        if self.show_confidences:
            label_parts.append(f"{track.confidence:.2f}")
        
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
    
    def draw_tracks(
        self,
        image: np.ndarray,
        tracks: List[Track],
        draw_history: bool = True
    ) -> np.ndarray:
        """Draw multiple tracks on image"""
        result = image.copy()
        for track in tracks:
            result = self.draw_track(result, track, draw_history=draw_history)
        return result

