"""
ROI (Region of Interest) and Zone detection module
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from .base import BaseModule
from ..data.detection import Detection
from ..data.track import Track
from ..data.roi import ROI


class ROIDetector(BaseModule):
    """ROI and zone detection module"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize ROI detector
        
        Args:
            config: Configuration dictionary with keys:
                - rois: List of ROI definitions
                - check_center: Check bbox center instead of overlap (default: True)
                - min_overlap: Minimum overlap ratio for bbox (default: 0.5)
        """
        super().__init__(config)
        self.rois: List[ROI] = []
        self.check_center = self.config.get("check_center", True)
        self.min_overlap = self.config.get("min_overlap", 0.5)
        self._load_rois()
    
    def _load_rois(self):
        """Load ROIs from configuration"""
        rois_config = self.config.get("rois", [])
        for roi_config in rois_config:
            roi = ROI(
                name=roi_config.get("name", f"roi_{len(self.rois)}"),
                points=roi_config.get("points", []),
                roi_type=roi_config.get("type", "polygon")
            )
            self.rois.append(roi)
    
    def add_roi(
        self,
        name: str,
        points: List[Tuple[float, float]],
        roi_type: str = "polygon"
    ):
        """
        Add a new ROI
        
        Args:
            name: ROI name
            points: List of (x, y) points
            roi_type: Type of ROI
        """
        roi = ROI(name, points, roi_type)
        self.rois.append(roi)
    
    def initialize(self) -> bool:
        """Initialize ROI detector"""
        self.is_initialized = True
        return True
    
    def check_detection_in_roi(
        self,
        detection: Detection,
        roi_name: Optional[str] = None
    ) -> bool:
        """
        Check if detection is in ROI
        
        Args:
            detection: Detection object
            roi_name: Specific ROI name to check (None for all)
            
        Returns:
            bool: True if in ROI
        """
        rois_to_check = [r for r in self.rois if roi_name is None or r.name == roi_name]
        
        if self.check_center:
            return any(roi.contains_bbox(detection.bbox) for roi in rois_to_check)
        else:
            # Check overlap
            for roi in rois_to_check:
                if self._bbox_overlaps_roi(detection.bbox, roi) >= self.min_overlap:
                    return True
            return False
    
    def check_track_in_roi(
        self,
        track: Track,
        roi_name: Optional[str] = None
    ) -> bool:
        """
        Check if track is in ROI
        
        Args:
            track: Track object
            roi_name: Specific ROI name to check (None for all)
            
        Returns:
            bool: True if in ROI
        """
        return self.check_detection_in_roi(
            type('Detection', (), {'bbox': track.bbox})(),
            roi_name
        )
    
    def filter_detections_by_roi(
        self,
        detections: List[Detection],
        roi_name: Optional[str] = None
    ) -> List[Detection]:
        """
        Filter detections that are in ROI
        
        Args:
            detections: List of Detection objects
            roi_name: Specific ROI name to filter (None for all)
            
        Returns:
            Filtered list of detections
        """
        return [
            det for det in detections
            if self.check_detection_in_roi(det, roi_name)
        ]
    
    def filter_tracks_by_roi(
        self,
        tracks: List[Track],
        roi_name: Optional[str] = None
    ) -> List[Track]:
        """
        Filter tracks that are in ROI
        
        Args:
            tracks: List of Track objects
            roi_name: Specific ROI name to filter (None for all)
            
        Returns:
            Filtered list of tracks
        """
        return [
            track for track in tracks
            if self.check_track_in_roi(track, roi_name)
        ]
    
    def get_detections_by_roi(
        self,
        detections: List[Detection]
    ) -> Dict[str, List[Detection]]:
        """
        Group detections by ROI
        
        Args:
            detections: List of Detection objects
            
        Returns:
            Dictionary mapping ROI names to detections
        """
        result = {roi.name: [] for roi in self.rois}
        result["none"] = []
        
        for det in detections:
            found = False
            for roi in self.rois:
                if self.check_detection_in_roi(det, roi.name):
                    result[roi.name].append(det)
                    found = True
                    break
            if not found:
                result["none"].append(det)
        
        return result
    
    def _bbox_overlaps_roi(
        self,
        bbox: Tuple[float, float, float, float],
        roi: ROI
    ) -> float:
        """Calculate overlap ratio between bbox and ROI"""
        # Simple approximation: check if bbox center is in ROI
        # For more accurate calculation, would need to compute intersection area
        if roi.contains_bbox(bbox):
            return 1.0
        return 0.0
    
    def process(
        self,
        detections: List[Detection],
        roi_name: Optional[str] = None
    ) -> List[Detection]:
        """
        Process detections and filter by ROI
        
        Args:
            detections: List of Detection objects
            roi_name: Specific ROI name to filter (None for all)
            
        Returns:
            Filtered detections
        """
        if not self.is_initialized:
            self.initialize()
        
        return self.filter_detections_by_roi(detections, roi_name)
    
    def process_batch(
        self,
        detections_list: List[List[Detection]],
        roi_name: Optional[str] = None
    ) -> List[List[Detection]]:
        """
        Filter multiple detection lists by ROI.
        
        Args:
            detections_list: List of detection lists (one per frame/image)
            roi_name: Specific ROI name to filter (None for all)
        
        Returns:
            List of filtered detection lists
        
        Example:
            ```python
            roi_detector = ROIDetector()
            roi_detector.initialize()
            
            # Process detections from multiple frames
            detections_batch = [detections_frame1, detections_frame2, detections_frame3]
            filtered_batch = roi_detector.process_batch(detections_batch)
            ```
        """
        if not self.is_initialized:
            self.initialize()
        
        return [
            self.filter_detections_by_roi(detections, roi_name)
            for detections in detections_list
        ]
    
    def get_rois(self) -> List[ROI]:
        """Get all ROIs"""
        return self.rois.copy()
    
    def get_roi_by_name(self, name: str) -> Optional[ROI]:
        """Get ROI by name"""
        for roi in self.rois:
            if roi.name == name:
                return roi
        return None

