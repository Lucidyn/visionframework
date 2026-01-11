"""
Counting module for objects in regions
"""

from typing import List, Dict, Any, Optional, Set
from collections import defaultdict
from .base import BaseModule
from ..data.detection import Detection
from ..data.track import Track
from .roi_detector import ROIDetector
from ..data.roi import ROI


class Counter(BaseModule):
    """Object counter for regions and zones"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize counter
        
        Args:
            config: Configuration dictionary with keys:
                - roi_detector: ROIDetector instance or config
                - count_entering: Count objects entering ROI (default: True)
                - count_exiting: Count objects exiting ROI (default: True)
                - count_inside: Count objects inside ROI (default: True)
                - track_direction: Track direction of movement (default: False)
        """
        super().__init__(config)
        self.roi_detector = None
        self.count_entering = self.config.get("count_entering", True)
        self.count_exiting = self.config.get("count_exiting", True)
        self.count_inside = self.config.get("count_inside", True)
        self.track_direction = self.config.get("track_direction", False)
        
        # Counters
        self.counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "entering": 0,
            "exiting": 0,
            "inside": 0,
            "total": 0
        })
        
        # Track previous states
        self.previous_tracks_in_roi: Dict[str, Set[int]] = defaultdict(set)
        self.track_directions: Dict[int, str] = {}  # Track ID -> direction
    
    def _initialize_roi_detector(self):
        """Initialize ROI detector"""
        roi_config = self.config.get("roi_detector")
        if isinstance(roi_config, ROIDetector):
            self.roi_detector = roi_config
        elif isinstance(roi_config, dict):
            self.roi_detector = ROIDetector(roi_config)
        else:
            # Create default ROI detector
            self.roi_detector = ROIDetector()
    
    def initialize(self) -> bool:
        """Initialize counter"""
        self._initialize_roi_detector()
        if not self.roi_detector.initialize():
            return False
        
        self.is_initialized = True
        return True
    
    def set_roi_detector(self, roi_detector: ROIDetector):
        """Set ROI detector"""
        self.roi_detector = roi_detector
        if self.is_initialized:
            self.roi_detector.initialize()
    
    def count_tracks(
        self,
        tracks: List[Track],
        roi_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Count tracks in ROI
        
        Args:
            tracks: List of Track objects
            roi_name: Specific ROI name (None for all ROIs)
            
        Returns:
            Dictionary with counting results
        """
        if not self.is_initialized:
            if not self.initialize():
                return {}
        
        if roi_name:
            rois_to_process = [self.roi_detector.get_roi_by_name(roi_name)]
        else:
            rois_to_process = self.roi_detector.get_rois()
        
        results = {}
        
        for roi in rois_to_process:
            if roi is None:
                continue
            
            roi_name_key = roi.name
            
            # Get current tracks in ROI
            current_tracks_in_roi = {
                track.track_id
                for track in tracks
                if self.roi_detector.check_track_in_roi(track, roi_name_key)
            }
            
            previous_tracks = self.previous_tracks_in_roi[roi_name_key]
            
            # Count entering
            entering = current_tracks_in_roi - previous_tracks
            if self.count_entering:
                self.counts[roi_name_key]["entering"] += len(entering)
                self.counts[roi_name_key]["total"] += len(entering)
            
            # Count exiting
            exiting = previous_tracks - current_tracks_in_roi
            if self.count_exiting:
                self.counts[roi_name_key]["exiting"] += len(exiting)
            
            # Count inside
            if self.count_inside:
                self.counts[roi_name_key]["inside"] = len(current_tracks_in_roi)
            
            # Track directions
            if self.track_direction:
                for track_id in entering:
                    self.track_directions[track_id] = "entering"
                for track_id in exiting:
                    self.track_directions[track_id] = "exiting"
            
            # Update previous state
            self.previous_tracks_in_roi[roi_name_key] = current_tracks_in_roi.copy()
            
            results[roi_name_key] = {
                "entering": len(entering),
                "exiting": len(exiting),
                "inside": len(current_tracks_in_roi),
                "total_entered": self.counts[roi_name_key]["entering"],
                "total_exited": self.counts[roi_name_key]["exiting"],
                "current_inside": self.counts[roi_name_key]["inside"],
                "total_count": self.counts[roi_name_key]["total"]
            }
        
        return results
    
    def count_detections(
        self,
        detections: List[Detection],
        roi_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Count detections in ROI (simple counting, no tracking)
        
        Args:
            detections: List of Detection objects
            roi_name: Specific ROI name (None for all ROIs)
            
        Returns:
            Dictionary with counting results
        """
        if not self.is_initialized:
            if not self.initialize():
                return {}
        
        filtered = self.roi_detector.filter_detections_by_roi(detections, roi_name)
        
        if roi_name:
            roi_name_key = roi_name
        else:
            # Count for all ROIs
            grouped = self.roi_detector.get_detections_by_roi(detections)
            results = {}
            for roi_name_key, dets in grouped.items():
                if roi_name_key != "none":
                    results[roi_name_key] = {
                        "count": len(dets),
                        "detections": dets
                    }
            return results
        
        return {
            roi_name_key: {
                "count": len(filtered),
                "detections": filtered
            }
        }
    
    def process(
        self,
        tracks: List[Track],
        roi_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process tracks and count
        
        Args:
            tracks: List of Track objects
            roi_name: Specific ROI name (None for all ROIs)
            
        Returns:
            Counting results
        """
        return self.count_tracks(tracks, roi_name)
    
    def get_counts(self, roi_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current counts
        
        Args:
            roi_name: Specific ROI name (None for all)
            
        Returns:
            Counts dictionary
        """
        if roi_name:
            return self.counts.get(roi_name, {}).copy()
        return {k: v.copy() for k, v in self.counts.items()}
    
    def reset_counts(self, roi_name: Optional[str] = None):
        """
        Reset counts
        
        Args:
            roi_name: Specific ROI name (None for all)
        """
        if roi_name:
            if roi_name in self.counts:
                self.counts[roi_name] = {
                    "entering": 0,
                    "exiting": 0,
                    "inside": 0,
                    "total": 0
                }
            self.previous_tracks_in_roi[roi_name] = set()
        else:
            self.counts.clear()
            self.previous_tracks_in_roi.clear()
            self.track_directions.clear()
    
    def reset(self):
        """Reset counter state"""
        super().reset()
        self.reset_counts()

