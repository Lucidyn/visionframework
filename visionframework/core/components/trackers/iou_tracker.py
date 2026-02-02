"""
IoU-based tracker implementation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .base_tracker import BaseTracker
from visionframework.data.detection import Detection
from visionframework.data.track import Track

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class IOUTracker(BaseTracker):
    """Multi-object tracker using IoU-based matching"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize tracker
        
        Args:
            config: Configuration dictionary with keys:
                - max_age: Maximum frames a track can be missing (default: 30)
                - min_hits: Minimum hits to confirm a track (default: 3)
                - iou_threshold: IoU threshold for matching (default: 0.3)
                - use_kalman: Whether to use Kalman filter (default: False)
        """
        super().__init__(config)
        self.tracks: List[Track] = []
        self.next_id = 0
        self.max_age = self.config.get("max_age", 30)
        self.min_hits = self.config.get("min_hits", 3)
        self.iou_threshold = self.config.get("iou_threshold", 0.3)
        self.use_kalman = self.config.get("use_kalman", False)
    
    def initialize(self) -> bool:
        """Initialize the tracker"""
        self.tracks = []
        self.next_id = 0
        self.is_initialized = True
        return True
    
    def _calculate_iou(self, box1: Tuple[float, float, float, float],
                       box2: Tuple[float, float, float, float]) -> float:
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _associate_detections_to_tracks(
        self,
        detections: List[Detection],
        tracks: List[Track]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate detections to tracks using Hungarian algorithm"""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # Build cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = self._calculate_iou(track.bbox, det.bbox)
                cost_matrix[i, j] = 1.0 - iou
        
        # Use Hungarian algorithm if scipy available, else greedy matching
        if SCIPY_AVAILABLE:
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
            matches = []
            unmatched_dets = set(range(len(detections)))
            unmatched_tracks = set(range(len(tracks)))
            
            for t_idx, d_idx in zip(track_indices, det_indices):
                if cost_matrix[t_idx, d_idx] < (1.0 - self.iou_threshold):
                    matches.append((t_idx, d_idx))
                    unmatched_dets.discard(d_idx)
                    unmatched_tracks.discard(t_idx)
            
            return matches, list(unmatched_dets), list(unmatched_tracks)
        else:
            # Greedy matching
            matches = []
            unmatched_dets = set(range(len(detections)))
            unmatched_tracks = set(range(len(tracks)))
            
            pairs = []
            for i in range(len(tracks)):
                for j in range(len(detections)):
                    cost = cost_matrix[i, j]
                    if cost < (1.0 - self.iou_threshold):
                        pairs.append((cost, i, j))
            
            pairs.sort()
            
            for cost, i, j in pairs:
                if i in unmatched_tracks and j in unmatched_dets:
                    matches.append((i, j))
                    unmatched_tracks.remove(i)
                    unmatched_dets.remove(j)
            
            return matches, list(unmatched_dets), list(unmatched_tracks)
    
    def update(self, detections: List[Detection], image=None) -> List[Track]:
        """Update tracker with new detections"""
        if not self.is_initialized:
            self.initialize()
        
        # Predict tracks
        for track in self.tracks:
            track.predict()
        
        # Associate detections to tracks
        matches, unmatched_dets, unmatched_tracks = self._associate_detections_to_tracks(
            detections, self.tracks
        )
        
        # Update matched tracks
        for track_idx, det_idx in matches:
            det = detections[det_idx]
            self.tracks[track_idx].update(det.bbox, det.confidence)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_track = Track(
                track_id=self.next_id,
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name
            )
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove old tracks (in-place removal to preserve track ages)
        # Keep tracks that either:
        # 1. Were just updated/matched (time_since_update == 0)
        # 2. Haven't exceeded max_age
        i = 0
        while i < len(self.tracks):
            track = self.tracks[i]
            if track.time_since_update > self.max_age:
                self.tracks.pop(i)
            else:
                i += 1
        
        # Return confirmed tracks (those with sufficient hits)
        return [track for track in self.tracks if track.age >= self.min_hits]
    
    def reset(self):
        """Reset tracker state"""
        super().reset()
        self.tracks = []
        self.next_id = 0
    
    def get_tracks(self) -> List[Track]:
        """Get all active tracks"""
        return self.tracks.copy()
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID"""
        for track in self.tracks:
            if track.track_id == track_id:
                return track
        return None

