"""
Base tracker interface
"""

from typing import List, Dict, Any, Optional, Tuple
from visionframework.core.base import BaseModule
from visionframework.data.detection import Detection
from visionframework.data.track import Track


class BaseTracker(BaseModule):
    """
    Base tracker class that all tracker implementations should inherit from
    
    This class provides the basic interface and functionality that all trackers
    should implement.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the tracker
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.tracks: List[Track] = []
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize the tracker
        
        Returns:
            bool: True if initialization was successful
        """
        self.tracks = []
        self.is_initialized = True
        return True
    
    def update(self, detections: List[Detection], frame: Optional[Any] = None) -> List[Track]:
        """
        Update the tracker with new detections
        
        Args:
            detections: List of detections
            frame: Optional frame for visual tracking or additional processing
        
        Returns:
            List of updated tracks
        """
        raise NotImplementedError("update() method must be implemented by subclass")
    
    def reset(self):
        """
        Reset the tracker to its initial state
        """
        self.tracks = []
        self.is_initialized = False
    
    def process(self, detections: List[Detection], frame: Optional[Any] = None) -> List[Track]:
        """
        Process detections using the update method
        
        Args:
            detections: List of detections
            frame: Optional frame for visual tracking or additional processing
        
        Returns:
            List of updated tracks
        """
        return self.update(detections, frame)
    
    def get_tracks(self) -> List[Track]:
        """
        Get current tracks
        
        Returns:
            List of current tracks
        """
        return self.tracks
