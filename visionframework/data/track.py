"""
Track data structures
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


class Track:
    """Track object container"""
    
    def __init__(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        confidence: float,
        class_id: int,
        class_name: Optional[str] = None
    ):
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.age = 0
        self.time_since_update = 0
        self.history = [bbox]
    
    def update(self, bbox: Tuple[float, float, float, float], confidence: float):
        """Update track with new detection"""
        self.bbox = bbox
        self.confidence = confidence
        self.time_since_update = 0
        self.history.append(bbox)
        if len(self.history) > 30:  # Keep last 30 positions
            self.history.pop(0)
    
    def predict(self):
        """Predict next position (simple linear prediction)"""
        self.age += 1
        self.time_since_update += 1
    
    def __repr__(self):
        return f"Track(id={self.track_id}, class={self.class_name}, conf={self.confidence:.2f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "age": self.age,
            "time_since_update": self.time_since_update
        }


class STrack:
    """Single track for ByteTrack"""
    
    def __init__(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        score: float,
        class_id: int,
        class_name: Optional[str] = None
    ):
        self.track_id = track_id
        self.bbox = bbox
        self.score = score
        self.confidence = score  # Alias for compatibility
        self.class_id = class_id
        self.class_name = class_name
        
        # Appearance features
        self.embedding: Optional[np.ndarray] = None
        
        # Track state
        self.is_activated = False
        self.state = "New"  # New, Tracked, Lost, Removed
        self.frame_id = 0
        self.start_frame = 0
        
        # Track history
        self.history = []
        self.update_bbox(bbox, score)
        
        # Kalman filter state (simplified)
        self.mean = np.array([bbox[0], bbox[1], bbox[2], bbox[3]], dtype=np.float32)
        self.covariance = np.eye(4, dtype=np.float32) * 2
    
    def update_bbox(self, bbox: Tuple[float, float, float, float], score: float):
        """Update bounding box"""
        self.bbox = bbox
        self.score = score
        self.confidence = score
        self.history.append(bbox)
        if len(self.history) > 30:
            self.history.pop(0)
    
    def activate(self, frame_id: int):
        """Activate track"""
        self.is_activated = True
        self.state = "Tracked"
        if self.frame_id == 0:
            self.start_frame = frame_id
        self.frame_id = frame_id
    
    def re_activate(self, new_track, frame_id: int, new_id: bool = False):
        """Re-activate track"""
        self.update_bbox(new_track.bbox, new_track.score)
        self.activate(frame_id)
        self.state = "Tracked"
        if new_id:
            self.track_id = new_track.track_id
    
    def update(self, new_track, frame_id: int):
        """Update track"""
        self.frame_id = frame_id
        self.update_bbox(new_track.bbox, new_track.score)
        self.state = "Tracked"
        self.is_activated = True
    
    def mark_lost(self):
        """Mark track as lost"""
        self.state = "Lost"
    
    def mark_removed(self):
        """Mark track as removed"""
        self.state = "Removed"
    
    @property
    def tlbr(self) -> Tuple[float, float, float, float]:
        """Get top-left bottom-right format"""
        return self.bbox
    
    @property
    def tlwh(self) -> Tuple[float, float, float, float]:
        """Get top-left width-height format"""
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)
    
    def to_xyah(self) -> Tuple[float, float, float, float]:
        """Convert to (x, y, aspect ratio, height) format"""
        x1, y1, x2, y2 = self.bbox
        w = x2 - x1
        h = y2 - y1
        x = x1 + w / 2
        y = y1 + h / 2
        a = w / h if h > 0 else 0
        return (x, y, a, h)
    
    def to_track(self):
        """Convert STrack to Track format for compatibility"""
        from .track import Track
        track = Track(
            track_id=self.track_id,
            bbox=self.bbox,
            confidence=self.score,
            class_id=self.class_id,
            class_name=self.class_name
        )
        track.history = self.history.copy()
        track.age = self.frame_id - self.start_frame if self.start_frame > 0 else self.frame_id
        return track
    
    def __repr__(self):
        return f"STrack(id={self.track_id}, state={self.state}, score={self.score:.2f})"

