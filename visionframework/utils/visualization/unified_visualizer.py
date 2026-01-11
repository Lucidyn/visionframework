"""
Unified visualizer combining all visualization capabilities
"""

import numpy as np
from typing import List, Optional, Dict, Any
from .base_visualizer import BaseVisualizer
from .detection_visualizer import DetectionVisualizer
from .track_visualizer import TrackVisualizer
from .pose_visualizer import PoseVisualizer
from ...data.detection import Detection
from ...data.track import Track
from ...data.pose import Pose


class Visualizer(BaseVisualizer):
    """Unified visualizer for all vision results"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.detection_viz = DetectionVisualizer(config)
        self.track_viz = TrackVisualizer(config)
        self.pose_viz = PoseVisualizer(config)
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection]
    ) -> np.ndarray:
        """Draw detections"""
        return self.detection_viz.draw_detections(image, detections)
    
    def draw_tracks(
        self,
        image: np.ndarray,
        tracks: List[Track],
        draw_history: bool = True
    ) -> np.ndarray:
        """Draw tracks"""
        return self.track_viz.draw_tracks(image, tracks, draw_history)
    
    def draw_poses(
        self,
        image: np.ndarray,
        poses: List[Pose],
        draw_skeleton: bool = True,
        draw_keypoints: bool = True,
        draw_bbox: bool = True
    ) -> np.ndarray:
        """Draw poses"""
        return self.pose_viz.draw_poses(image, poses, draw_skeleton, draw_keypoints, draw_bbox)
    
    def draw_results(
        self,
        image: np.ndarray,
        detections: Optional[List[Detection]] = None,
        tracks: Optional[List[Track]] = None,
        poses: Optional[List[Pose]] = None,
        draw_history: bool = True
    ) -> np.ndarray:
        """Draw all results on image"""
        result = image.copy()
        
        # Draw tracks first (they have history trails)
        if tracks:
            result = self.draw_tracks(result, tracks, draw_history=draw_history)
        
        # Draw poses
        if poses:
            result = self.draw_poses(result, poses)
        
        # Draw detections if no tracks (or if explicitly requested)
        if detections and not tracks:
            result = self.draw_detections(result, detections)
        
        return result

