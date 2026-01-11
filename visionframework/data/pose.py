"""
Pose data structures
"""

from typing import List, Dict, Any, Optional


class KeyPoint:
    """Keypoint container"""
    
    def __init__(
        self,
        x: float,
        y: float,
        confidence: float,
        keypoint_id: int,
        keypoint_name: Optional[str] = None
    ):
        self.x = x
        self.y = y
        self.confidence = confidence
        self.keypoint_id = keypoint_id
        self.keypoint_name = keypoint_name
    
    def __repr__(self):
        return f"KeyPoint(id={self.keypoint_id}, name={self.keypoint_name}, conf={self.confidence:.2f}, pos=({self.x:.1f}, {self.y:.1f}))"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "x": self.x,
            "y": self.y,
            "confidence": self.confidence,
            "keypoint_id": self.keypoint_id,
            "keypoint_name": self.keypoint_name
        }


class Pose:
    """Pose container"""
    
    def __init__(
        self,
        bbox: tuple,
        keypoints: List[KeyPoint],
        confidence: float,
        pose_id: Optional[int] = None
    ):
        self.bbox = bbox
        self.keypoints = keypoints
        self.confidence = confidence
        self.pose_id = pose_id
    
    def __repr__(self):
        return f"Pose(id={self.pose_id}, conf={self.confidence:.2f}, keypoints={len(self.keypoints)})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "bbox": self.bbox,
            "keypoints": [kp.to_dict() for kp in self.keypoints],
            "confidence": self.confidence,
            "pose_id": self.pose_id
        }
    
    def get_keypoint_by_id(self, keypoint_id: int) -> Optional[KeyPoint]:
        """Get keypoint by ID"""
        for kp in self.keypoints:
            if kp.keypoint_id == keypoint_id:
                return kp
        return None
    
    def get_keypoint_by_name(self, keypoint_name: str) -> Optional[KeyPoint]:
        """Get keypoint by name"""
        for kp in self.keypoints:
            if kp.keypoint_name == keypoint_name:
                return kp
        return None

