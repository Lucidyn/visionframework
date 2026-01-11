"""
ReID Tracker implementation
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from scipy.spatial.distance import cdist
from .base_tracker import BaseTracker
from ...data.track import STrack
from ...data.detection import Detection
from ..reid import ReIDExtractor
from ...utils.logger import get_logger

logger = get_logger(__name__)

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ReIDTracker(BaseTracker):
    """
    Tracker using ReID features for association (simplified DeepSORT-like)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ReID Tracker
        
        Args:
            config: Configuration dictionary with keys:
                - max_age: Maximum frames to keep lost track (default: 30)
                - min_hits: Minimum hits to activate track (default: 3)
                - iou_threshold: IoU threshold for matching (default: 0.3)
                - reid_weight: Weight for ReID distance (0.0 to 1.0) (default: 0.7)
                - reid_config: Config for ReIDExtractor (dict)
        """
        super().__init__(config)
        self.max_age = self.config.get("max_age", 30)
        self.min_hits = self.config.get("min_hits", 3)
        self.iou_threshold = self.config.get("iou_threshold", 0.3)
        self.reid_weight = self.config.get("reid_weight", 0.7)
        self.reid_config = self.config.get("reid_config", {})
        
        self.reid_extractor = ReIDExtractor(self.reid_config)
        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []
        
        self.frame_id = 0
        self.next_id = 1
        
    def initialize(self) -> bool:
        """Initialize tracker and ReID model"""
        if not self.reid_extractor.initialize():
            logger.warning("Failed to initialize ReID extractor")
            return False
        
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0
        self.next_id = 1
        self.is_initialized = True
        return True

    def _calculate_iou(self, box1, box2):
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

    def _get_cost_matrix(self, tracks: List[STrack], detections: List[Detection], embeddings: np.ndarray) -> np.ndarray:
        """
        Calculate cost matrix combining IoU and ReID distance
        
        Cost = reid_weight * reid_dist + (1 - reid_weight) * iou_dist
        """
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)))
            
        # IoU distance (1 - IoU)
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = 1.0 - self._calculate_iou(track.bbox, det.bbox)
                
        # ReID distance (Cosine distance)
        # embeddings shape: (N_dets, 2048)
        # track embeddings
        track_embs = []
        valid_track_indices = []
        for i, track in enumerate(tracks):
            if track.embedding is not None:
                track_embs.append(track.embedding)
                valid_track_indices.append(i)
            else:
                # Handle tracks without embedding (should happen rarely if initialized correctly)
                # Maybe fill with zeros or skip ReID for them?
                # For simplicity, we treat them as having max distance
                pass
                
        reid_matrix = np.ones((len(tracks), len(detections))) # Default max distance
        
        if len(track_embs) > 0 and len(embeddings) > 0:
            track_embs = np.stack(track_embs)
            # cdist returns distance matrix
            dists = cdist(track_embs, embeddings, metric='cosine')
            # Map back to full matrix
            for k, track_idx in enumerate(valid_track_indices):
                reid_matrix[track_idx, :] = dists[k, :]
                
        # Combine
        # If tracks have no embedding, they rely purely on IoU (since ReID dist is 1.0)
        # But we want to avoid matching if IoU is too low regardless of ReID
        
        cost_matrix = self.reid_weight * reid_matrix + (1 - self.reid_weight) * iou_matrix
        return cost_matrix

    def update(self, detections: List[Detection], image: Optional[np.ndarray] = None) -> List[STrack]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of Detection objects
            image: Current frame image (required for ReID extraction)
        """
        if not self.is_initialized:
            self.initialize()
            
        self.frame_id += 1
        
        # 1. Extract ReID features
        embeddings = np.empty((0, 2048))
        if image is not None and len(detections) > 0:
            bboxes = [d.bbox for d in detections]
            embeddings = self.reid_extractor.extract(image, bboxes)
        
        # 2. Predict (skip for now as STrack KF is placeholder)
        
        # 3. Match
        # Pool all candidate tracks (tracked + lost)
        # In simple DeepSORT, we might prioritize tracked, then lost.
        # Here we pool them for simplicity or match tracked first.
        # Let's match all confirmed tracks.
        
        confirmed_tracks = [t for t in self.tracked_tracks if t.is_activated]
        # Add lost tracks that are not too old
        candidate_tracks = confirmed_tracks + [t for t in self.lost_tracks if t.state != "Removed"]
        
        # Match
        if len(candidate_tracks) > 0 and len(detections) > 0:
            cost_matrix = self._get_cost_matrix(candidate_tracks, detections, embeddings)
            
            if SCIPY_AVAILABLE:
                row_indices, col_indices = linear_sum_assignment(cost_matrix)
            else:
                # Fallback greedy matching
                row_indices, col_indices = [], []
                # Copy matrix to modify it
                cost_copy = cost_matrix.copy()
                num_rows, num_cols = cost_copy.shape
                iter_limit = min(num_rows, num_cols)
                
                for _ in range(iter_limit):
                    # Find min
                    min_idx = np.argmin(cost_copy)
                    r, c = np.unravel_index(min_idx, (num_rows, num_cols))
                    
                    if cost_copy[r, c] == np.inf:
                        break
                        
                    row_indices.append(r)
                    col_indices.append(c)
                    
                    # Mask row and col
                    cost_copy[r, :] = np.inf
                    cost_copy[:, c] = np.inf
                
            matches = []
            unmatched_tracks = set(range(len(candidate_tracks)))
            unmatched_detections = set(range(len(detections)))
            
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < 1.0: # Threshold?
                    matches.append((row, col))
                    unmatched_tracks.discard(row)
                    unmatched_detections.discard(col)
                    
            # Update matched tracks
            for track_idx, det_idx in matches:
                track = candidate_tracks[track_idx]
                det = detections[det_idx]
                emb = embeddings[det_idx] if len(embeddings) > det_idx else None
                
                track.update_bbox(det.bbox, det.confidence)
                if emb is not None:
                    # Update embedding (EMA)
                    if track.embedding is None:
                        track.embedding = emb
                    else:
                        alpha = 0.9
                        track.embedding = alpha * track.embedding + (1 - alpha) * emb
                        track.embedding /= np.linalg.norm(track.embedding)
                
                track.frame_id = self.frame_id
                if track in self.lost_tracks:
                    self.lost_tracks.remove(track)
                    self.tracked_tracks.append(track)
                    track.state = "Tracked"
            
            # Create new tracks
            for det_idx in unmatched_detections:
                det = detections[det_idx]
                emb = embeddings[det_idx] if len(embeddings) > det_idx else None
                
                new_track = STrack(self.next_id, det.bbox, det.confidence, det.class_id, det.class_name)
                new_track.embedding = emb
                new_track.frame_id = self.frame_id
                new_track.start_frame = self.frame_id
                new_track.state = "New" # Wait for hits
                # For simplicity, activate immediately if conf is high enough?
                # Standard logic: New -> (hits) -> Activated
                # Here simplified:
                if det.confidence > 0.6: # Configurable
                    new_track.is_activated = True
                    new_track.state = "Tracked"
                    self.tracked_tracks.append(new_track)
                    self.next_id += 1
                
            # Handle unmatched tracks
            for track_idx in unmatched_tracks:
                track = candidate_tracks[track_idx]
                if track.state == "Tracked":
                    track.state = "Lost"
                    if track in self.tracked_tracks:
                        self.tracked_tracks.remove(track)
                        self.lost_tracks.append(track)
                
        else:
            # No matching possible
            # Create all as new
            for i, det in enumerate(detections):
                emb = embeddings[i] if len(embeddings) > i else None
                new_track = STrack(self.next_id, det.bbox, det.confidence, det.class_id, det.class_name)
                new_track.embedding = emb
                new_track.frame_id = self.frame_id
                new_track.start_frame = self.frame_id
                if det.confidence > 0.6:
                    new_track.is_activated = True
                    new_track.state = "Tracked"
                    self.tracked_tracks.append(new_track)
                    self.next_id += 1
            
            # Mark all tracks as lost
            for track in self.tracked_tracks:
                track.state = "Lost"
                self.lost_tracks.append(track)
            self.tracked_tracks = []

        # Remove old lost tracks
        self.lost_tracks = [t for t in self.lost_tracks if self.frame_id - t.frame_id < self.max_age]
        
        return self.tracked_tracks
