"""
ByteTrack tracker implementation
ByteTrack: Multi-Object Tracking by Associating Every Detection Box
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .base_tracker import BaseTracker
from ...data.track import STrack
from ...data.detection import Detection

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ByteTracker(BaseTracker):
    """ByteTrack multi-object tracker"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ByteTracker
        
        Args:
            config: Configuration dictionary with keys:
                - track_thresh: Detection confidence threshold (default: 0.5)
                - track_buffer: Buffer for lost tracks (default: 30)
                - match_thresh: Matching threshold (default: 0.8)
                - frame_rate: Frame rate for tracking (default: 30)
                - min_box_area: Minimum box area (default: 10)
        """
        super().__init__(config)
        self.track_thresh = self.config.get("track_thresh", 0.5)
        self.track_buffer = self.config.get("track_buffer", 30)
        self.match_thresh = self.config.get("match_thresh", 0.8)
        self.frame_rate = self.config.get("frame_rate", 30)
        self.min_box_area = self.config.get("min_box_area", 10)
        
        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []
        self.frame_id = 0
        self.next_id = 1
    
    def initialize(self) -> bool:
        """Initialize tracker"""
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self.next_id = 1
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
    
    def _iou_distance(self, tracks: List[STrack], detections: List[Detection]) -> np.ndarray:
        """Calculate IoU distance matrix"""
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)), dtype=np.float32)
        
        cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = self._calculate_iou(track.bbox, det.bbox)
                cost_matrix[i, j] = 1.0 - iou
        
        return cost_matrix
    
    def _linear_assignment(self, cost_matrix: np.ndarray, thresh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Linear assignment using Hungarian algorithm"""
        if cost_matrix.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
        
        if SCIPY_AVAILABLE:
            matched_indices = linear_sum_assignment(cost_matrix)
            matched_indices = np.array(list(zip(*matched_indices)))
        else:
            matched_indices = []
            used_rows = set()
            used_cols = set()
            
            for i in range(cost_matrix.shape[0]):
                for j in range(cost_matrix.shape[1]):
                    if i not in used_rows and j not in used_cols:
                        if cost_matrix[i, j] < thresh:
                            matched_indices.append([i, j])
                            used_rows.add(i)
                            used_cols.add(j)
            matched_indices = np.array(matched_indices) if matched_indices else np.empty((0, 2), dtype=int)
        
        unmatched_a = []
        unmatched_b = []
        
        for i, cost in enumerate(cost_matrix):
            if i not in matched_indices[:, 0]:
                unmatched_a.append(i)
        
        for i in range(cost_matrix.shape[1]):
            if i not in matched_indices[:, 1]:
                unmatched_b.append(i)
        
        return matched_indices, np.array(unmatched_a), np.array(unmatched_b)
    
    def update(self, detections, image=None) -> List[STrack]:
        """Process detections and update tracks"""
        if not self.is_initialized:
            self.initialize()
        
        self.frame_id += 1
        
        # Separate detections by confidence
        detections_high = [d for d in detections if d.confidence >= self.track_thresh]
        detections_low = [d for d in detections if d.confidence < self.track_thresh]
        
        # Filter by minimum area
        detections_high = [
            d for d in detections_high
            if (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) >= self.min_box_area
        ]
        
        # Update tracked tracks
        activated_tracks = []
        refined_tracks = []
        lost_tracks = []
        removed_tracks = []
        
        # Match high confidence detections with tracked tracks
        if len(self.tracked_tracks) > 0:
            cost_matrix = self._iou_distance(self.tracked_tracks, detections_high)
            matches, u_track, u_detection = self._linear_assignment(cost_matrix, 1.0 - self.match_thresh)
            
            for itrack, idet in matches:
                track = self.tracked_tracks[itrack]
                det = detections_high[idet]
                if track.state == "Tracked":
                    track.update(det, self.frame_id)
                    activated_tracks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refined_tracks.append(track)
            
            for itrack in u_track:
                track = self.tracked_tracks[itrack]
                if track.state != "Lost":
                    track.mark_lost()
                    lost_tracks.append(track)
        
        # Match low confidence detections with lost tracks
        if len(self.lost_tracks) > 0:
            cost_matrix = self._iou_distance(self.lost_tracks, detections_low)
            matches, u_lost, u_detection_low = self._linear_assignment(cost_matrix, 1.0 - self.match_thresh)
            
            for itrack, idet in matches:
                track = self.lost_tracks[itrack]
                det = detections_low[idet]
                track.re_activate(det, self.frame_id, new_id=False)
                refined_tracks.append(track)
        
        # Create new tracks for unmatched high confidence detections
        for det in detections_high:
            # Check if already matched
            matched = False
            for match in matches if len(self.tracked_tracks) > 0 else []:
                if det == detections_high[match[1]]:
                    matched = True
                    break
            if not matched:
                track = STrack(
                    track_id=self.next_id,
                    bbox=det.bbox,
                    score=det.confidence,
                    class_id=det.class_id,
                    class_name=det.class_name
                )
                track.activate(self.frame_id)
                activated_tracks.append(track)
                self.next_id += 1
        
        # Update track lists
        self.tracked_tracks = [t for t in activated_tracks if t.state == "Tracked"]
        self.tracked_tracks.extend(refined_tracks)
        
        # Update lost tracks
        self.lost_tracks = [t for t in self.lost_tracks if t.state == "Lost"]
        self.lost_tracks.extend(lost_tracks)
        self.lost_tracks = [t for t in self.lost_tracks if self.frame_id - t.frame_id <= self.track_buffer]
        
        # Remove old tracks
        removed_tracks = [t for t in self.lost_tracks if self.frame_id - t.frame_id > self.track_buffer]
        for t in removed_tracks:
            t.mark_removed()
        self.removed_tracks.extend(removed_tracks)
        self.lost_tracks = [t for t in self.lost_tracks if t.state != "Removed"]
        
        # Return active tracks
        return [t for t in self.tracked_tracks if t.is_activated]
    
    def reset(self):
        """Reset tracker state"""
        super().reset()
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self.next_id = 1

