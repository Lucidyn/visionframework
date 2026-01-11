"""
Pose visualization
"""

import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from .base_visualizer import BaseVisualizer
from ...data.pose import Pose


class PoseVisualizer(BaseVisualizer):
    """Visualizer for pose results"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.show_labels = self.config.get("show_labels", True)
        self.show_confidences = self.config.get("show_confidences", True)
    
    def draw_pose(
        self,
        image: np.ndarray,
        pose: Pose,
        draw_skeleton: bool = True,
        draw_keypoints: bool = True,
        draw_bbox: bool = True
    ) -> np.ndarray:
        """Draw a single pose on image"""
        result = image.copy()
        
        # Draw bounding box
        if draw_bbox:
            x1, y1, x2, y2 = map(int, pose.bbox)
            color = self._get_color(pose.pose_id if pose.pose_id is not None else 0)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, self.line_thickness)
            
            # Draw confidence
            if self.show_confidences:
                label = f"Pose {pose.pose_id}: {pose.confidence:.2f}" if pose.pose_id is not None else f"{pose.confidence:.2f}"
                cv2.putText(result, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, color, self.line_thickness)
        
        # Draw keypoints
        if draw_keypoints:
            for kp in pose.keypoints:
                x, y = int(kp.x), int(kp.y)
                color = self._get_color(kp.keypoint_id)
                cv2.circle(result, (x, y), 5, color, -1)
                
                if self.show_labels and kp.keypoint_name:
                    cv2.putText(result, kp.keypoint_name, (x + 5, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.6,
                               (255, 255, 255), 1)
        
        # Draw skeleton
        if draw_skeleton and len(pose.keypoints) > 0:
            skeleton_connections = [
                ("nose", "left_eye"), ("nose", "right_eye"),
                ("left_eye", "left_ear"), ("right_eye", "right_ear"),
                ("left_shoulder", "right_shoulder"),
                ("left_shoulder", "left_hip"),
                ("right_shoulder", "right_hip"),
                ("left_hip", "right_hip"),
                ("left_shoulder", "left_elbow"),
                ("left_elbow", "left_wrist"),
                ("right_shoulder", "right_elbow"),
                ("right_elbow", "right_wrist"),
                ("left_hip", "left_knee"),
                ("left_knee", "left_ankle"),
                ("right_hip", "right_knee"),
                ("right_knee", "right_ankle"),
            ]
            
            for start_name, end_name in skeleton_connections:
                start_kp = pose.get_keypoint_by_name(start_name)
                end_kp = pose.get_keypoint_by_name(end_name)
                
                if start_kp and end_kp:
                    cv2.line(result,
                            (int(start_kp.x), int(start_kp.y)),
                            (int(end_kp.x), int(end_kp.y)),
                            (0, 255, 0), 2)
        
        return result
    
    def draw_poses(
        self,
        image: np.ndarray,
        poses: List[Pose],
        draw_skeleton: bool = True,
        draw_keypoints: bool = True,
        draw_bbox: bool = True
    ) -> np.ndarray:
        """Draw multiple poses on image"""
        result = image.copy()
        for pose in poses:
            result = self.draw_pose(result, pose, draw_skeleton, draw_keypoints, draw_bbox)
        return result

