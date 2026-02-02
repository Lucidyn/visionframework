"""
Base detector interface

This module defines the abstract base class for all detector implementations.
All detectors must inherit from BaseDetector and implement the detect() method.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np
from visionframework.core.base import BaseModule
from visionframework.data.detection import Detection


class BaseDetector(BaseModule, ABC):
    """
    Base class for all detectors
    
    This abstract class defines the interface that all detector implementations
    must follow. Subclasses should implement the detect() method to perform
    object detection on input images.
    
    Example:
        ```python
        class MyDetector(BaseDetector):
            def initialize(self) -> bool:
                # Initialize model
                return True
            
            def detect(self, image: np.ndarray) -> List[Detection]:
                # Perform detection
                return detections
        ```
    """
    
    @abstractmethod
    def detect(self, image: np.ndarray, categories: Optional[Union[list, tuple]] = None) -> List[Detection]:
        """
        Detect objects in image
        
        This method must be implemented by subclasses to perform object detection
        on the input image.
        
         Args:
             image: Input image in BGR format (OpenCV standard), numpy array with shape (H, W, 3).
                 Data type should be uint8 with values in range [0, 255].
             categories: Optional list/tuple of class ids or class names to keep.
                   If None, all detected classes are returned.
        
        Returns:
            List[Detection]: List of Detection objects, each containing:
                - bbox: Tuple of (x1, y1, x2, y2) coordinates
                - confidence: Confidence score between 0.0 and 1.0
                - class_id: Integer class ID
                - class_name: String class name
                - mask: Optional segmentation mask (if segmentation is supported)
            
            Returns empty list if no objects are detected or if an error occurs.
        
        Raises:
            RuntimeError: If detector is not initialized
            ValueError: If image is invalid (wrong format, shape, or data type)
        
        Note:
            Subclasses should check is_initialized before processing and return
            an empty list if not initialized.
        """
        pass
    
    def process(self, image: np.ndarray, categories: Optional[Union[list, tuple]] = None) -> List[Detection]:
        """
        Alias for detect method
        
        This method provides an alternative name for detect() to maintain
        consistency with the BaseModule interface.
        
        Args:
            image: Input image in BGR format (numpy array, shape: (H, W, 3))
        
        Returns:
            List[Detection]: List of detected objects
        """
        return self.detect(image, categories=categories)
    
    def detect_batch(self, images: List[np.ndarray], categories: Optional[Union[list, tuple]] = None) -> List[List[Detection]]:
        """
        Detect objects in multiple images in batch mode
        
        This method processes multiple images in a batch, which is more efficient
        than processing them individually. Subclasses can override this method
        to implement optimized batch processing.
        
        Args:
            images: List of input images in BGR format (numpy arrays, shape: (H, W, 3))
            categories: Optional list/tuple of class ids or class names to keep.
                   If None, all detected classes are returned.
        
        Returns:
            List[List[Detection]]: List of detection lists, one per image.
                                  Each inner list contains Detection objects for that image.
        
        Raises:
            RuntimeError: If detector is not initialized
            ValueError: If images are invalid (wrong format, shape, or data type)
        """
        if not images:
            return []
        
        # Default implementation: process each image individually
        results = []
        for image in images:
            detections = self.detect(image, categories=categories)
            results.append(detections)
        return results
    
    def process_batch(self, images: List[np.ndarray], categories: Optional[Union[list, tuple]] = None) -> List[List[Detection]]:
        """
        Alias for detect_batch method
        
        This method provides an alternative name for detect_batch() to maintain
        consistency with the BaseModule interface.
        
        Args:
            images: List of input images in BGR format (numpy arrays, shape: (H, W, 3))
            categories: Optional list/tuple of class ids or class names to keep.
                   If None, all detected classes are returned.
        
        Returns:
            List[List[Detection]]: List of detection lists, one per image.
                                  Each inner list contains Detection objects for that image.
        """
        return self.detect_batch(images, categories=categories)

