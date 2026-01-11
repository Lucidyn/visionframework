"""
Base tracker interface
"""

from abc import ABC, abstractmethod
from typing import List
from ...data.detection import Detection
from ..base import BaseModule


class BaseTracker(BaseModule, ABC):
    """Base class for all trackers"""
    
    @abstractmethod
    def update(self, detections: List[Detection], image=None):
        """
        Update tracker with new detections
        
        Args:
            detections: List of Detection objects
            image: Optional current frame (needed for ReID trackers)
            
        Returns:
            List of track objects
        """
        pass
    
    def process(self, detections: List[Detection], image=None):
        """Alias for update method"""
        return self.update(detections, image=image)

