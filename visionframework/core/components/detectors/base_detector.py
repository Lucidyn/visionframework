"""
Base detector interface

This module defines the abstract base class for all detector implementations.
All detectors must inherit from BaseDetector and implement the detect() method.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union, Iterator, Tuple, Any, Dict
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
    
    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Validate that *image* is a proper BGR numpy array.

        Raises:
            TypeError: if *image* is not an ndarray.
            ValueError: if shape or dtype is wrong.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected np.ndarray, got {type(image).__name__}")
        if image.ndim not in (2, 3):
            raise ValueError(f"Image must be 2D (grayscale) or 3D (H,W,C), got ndim={image.ndim}")
        if image.size == 0:
            raise ValueError("Image is empty (zero pixels)")

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

    def detect_source(
        self,
        source: Union[str, int, List[Union[str, int]], np.ndarray, Path],
        categories: Optional[Union[list, tuple]] = None,
        *,
        recursive_folder: bool = False,
        video_skip_frames: int = 0,
        video_start_frame: int = 0,
        video_end_frame: Optional[int] = None,
    ) -> Iterator[Tuple[np.ndarray, Dict[str, Any], List[Detection]]]:
        """
        Detect objects on a unified media source (image, video, stream, folder, or list).

        Yields (frame, meta, detections) for each frame. Accepts:
        - str: path to image, video, or folder
        - int: camera index (e.g. 0)
        - List[Union[str, int]]: multiple paths or camera indices
        - np.ndarray: single BGR image

        Args:
            source: Image path, video path/URL, camera index, folder path,
                    list of the above, or single BGR numpy array.
            categories: Optional list/tuple of class ids or names to keep.
            recursive_folder: If source is a folder, include subfolders.
            video_skip_frames: For video, skip this many frames between reads.
            video_start_frame: For video, start at this frame index.
            video_end_frame: For video, stop at this frame index (None = to end).

        Yields:
            (frame, meta, detections): frame (BGR), meta dict (source_path, frame_index, ...), list of Detection.
        """
        from visionframework.utils.io.media_source import iter_frames
        for frame, meta in iter_frames(
            source,
            recursive_folder=recursive_folder,
            video_skip_frames=video_skip_frames,
            video_start_frame=video_start_frame,
            video_end_frame=video_end_frame,
        ):
            detections = self.detect(frame, categories=categories)
            yield frame, meta, detections

