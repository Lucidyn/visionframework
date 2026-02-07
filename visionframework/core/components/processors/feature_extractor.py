"""
Base Feature Extractor

Abstract base class for all feature extractors.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import numpy as np


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.
    
    All feature extractors (CLIP, ReID, Pose, etc.) should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, model_name: str, device: str = "cpu"):
        """
        Initialize feature extractor.
        
        Args:
            model_name: Name of the model to use
            device: Device to run on ("cpu", "cuda", etc.)
        """
        self.model_name = model_name
        self.device = device
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the feature extractor and load model.
        
        Should be called before using extract().
        """
        pass
    
    @abstractmethod
    def extract(self, input_data: Any) -> Union[np.ndarray, dict]:
        """
        Extract features from input data.
        
        Args:
            input_data: Input data (image, text, pose data, etc.)
        
        Returns:
            Extracted features as numpy array or dictionary
        """
        pass
    
    @property
    def is_initialized(self) -> bool:
        """Check if extractor is initialized."""
        return self._initialized
    
    def to(self, device: str) -> None:
        """
        Move extractor to different device.
        
        Args:
            device: Device name ("cpu", "cuda", etc.)
        """
        self.device = device
        if self._initialized:
            self._move_to_device(device)
    
    @abstractmethod
    def _move_to_device(self, device: str) -> None:
        """
        Move internal model to device. Implemented by subclasses.
        
        Args:
            device: Device name
        """
        pass
