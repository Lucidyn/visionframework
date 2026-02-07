"""
ByteTrack tracker implementation
ByteTrack: Multi-Object Tracking by Associating Every Detection Box
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .base_tracker import BaseTracker
from .utils import iou_cost_matrix, linear_assignment
from visionframework.data.track import STrack
from visionframework.data.detection import Detection


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
    
    def _iou_distance(self, tracks: List[STrack], detections: List[Detection]) -> np.ndarray:
        """Calculate IoU distance matrix using shared utility."""
        return iou_cost_matrix(
            [t.bbox for t in tracks],
            [d.bbox for d in detections],
        )
    
    def update(self, detections, image=None) -> List[STrack]:
        """Process detections and update tracks"""
        detections = self._validate_detections(detections)
        if not self.is_initialized:
            self.initialize()
        
        self.frame_id += 1
        
        if not detections:
            # Age existing tracks even when there are no new detections
            for track in self.tracked_tracks:
                if track.state != "Lost":
                    track.mark_lost()
            self.lost_tracks.extend([t for t in self.tracked_tracks if t.state == "Lost"])
            self.tracked_tracks = []
            self.lost_tracks = [t for t in self.lost_tracks if self.frame_id - t.frame_id <= self.track_buffer]
            return []
        
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
        unmatched_det_indices: np.ndarray = np.arange(len(detections_high))
        if len(self.tracked_tracks) > 0:
            cost_matrix = self._iou_distance(self.tracked_tracks, detections_high)
            matches, u_track, u_detection = linear_assignment(cost_matrix, 1.0 - self.match_thresh)
            unmatched_det_indices = u_detection
            
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
            low_matches, u_lost, u_detection_low = linear_assignment(cost_matrix, 1.0 - self.match_thresh)
            
            for itrack, idet in low_matches:
                track = self.lost_tracks[itrack]
                det = detections_low[idet]
                track.re_activate(det, self.frame_id, new_id=False)
                refined_tracks.append(track)
        
        # Create new tracks for unmatched high confidence detections
        for idet in unmatched_det_indices:
            det = detections_high[idet]
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

