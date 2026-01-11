"""
Pipeline for combining detection and tracking

This module provides a complete vision processing pipeline that combines
object detection and tracking in a single, easy-to-use interface.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .base import BaseModule
from .detector import Detector
from .tracker import Tracker
from ..data.detection import Detection
from ..data.track import Track
from ..utils.logger import get_logger

logger = get_logger(__name__)


class VisionPipeline(BaseModule):
    """
    Complete vision pipeline combining detection and tracking
    
    This class provides a unified interface for processing images or video frames
    through both detection and tracking stages. It manages the detector and tracker
    instances internally and provides a simple process() method for end-to-end processing.
    
    Example:
        ```python
        # Create pipeline with detection and tracking
        pipeline = VisionPipeline({
            "enable_tracking": True,
            "detector_config": {
                "model_type": "yolo",
                "model_path": "yolov8n.pt",
                "conf_threshold": 0.25
            },
            "tracker_config": {
                "max_age": 30,
                "min_hits": 3
            }
        })
        pipeline.initialize()
        
        # Process frames
        results = pipeline.process(frame)
        detections = results["detections"]
        tracks = results["tracks"]
        ```
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize vision pipeline
        
        Args:
            config: Configuration dictionary with keys:
                - detector_config: Configuration dictionary for the detector.
                                  See Detector.__init__() for available parameters.
                                  If not provided, default detector configuration is used.
                - tracker_config: Configuration dictionary for the tracker.
                                 See Tracker.__init__() for available parameters.
                                 Only used if enable_tracking is True.
                - enable_tracking: Boolean flag to enable/disable tracking (default: True).
                                 If False, only detection is performed.
        
        Note:
            Configuration validation is performed automatically for both detector
            and tracker configurations if their validate_config() methods are implemented.
        """
        super().__init__(config)
        self.detector: Optional[Detector] = None
        self.tracker: Optional[Tracker] = None
        self.enable_tracking: bool = self.config.get("enable_tracking", True)
        self.detector_config: Dict[str, Any] = self.config.get("detector_config", {})
        self.tracker_config: Dict[str, Any] = self.config.get("tracker_config", {})
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate pipeline configuration
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            Tuple[bool, Optional[str]]: 
                - (True, None) if valid
                - (False, error_message) if invalid
        
        Validates:
            - enable_tracking is a boolean
            - detector_config is a dictionary (if provided)
            - tracker_config is a dictionary (if provided)
        """
        # Validate enable_tracking
        if "enable_tracking" in config:
            enable_tracking = config["enable_tracking"]
            if not isinstance(enable_tracking, bool):
                return False, f"enable_tracking must be a boolean, got {type(enable_tracking).__name__}"
        
        # Validate detector_config
        if "detector_config" in config:
            detector_config = config["detector_config"]
            if not isinstance(detector_config, dict):
                return False, f"detector_config must be a dictionary, got {type(detector_config).__name__}"
        
        # Validate tracker_config
        if "tracker_config" in config:
            tracker_config = config["tracker_config"]
            if not isinstance(tracker_config, dict):
                return False, f"tracker_config must be a dictionary, got {type(tracker_config).__name__}"
        
        return True, None
    
    def initialize(self) -> bool:
        """
        Initialize detector and tracker
        
        This method initializes both the detector and tracker (if enabled) components
        of the pipeline. It performs all necessary setup steps for end-to-end processing.
        
        Returns:
            bool: True if all components initialized successfully, False otherwise.
                  On failure, errors are logged with detailed information.
        
        Raises:
            ValueError: If configuration is invalid
            ImportError: If required dependencies are missing
            RuntimeError: If component initialization fails
        
        Note:
            Initialization sequence:
            1. Initialize detector (required)
            2. Initialize tracker (if enable_tracking is True)
            3. Set is_initialized flag if all components initialized successfully
        
            If any component fails to initialize, the entire pipeline initialization
            fails and is_initialized remains False.
        
        Example:
            ```python
            pipeline = VisionPipeline({
                "enable_tracking": True,
                "detector_config": {"model_path": "yolov8n.pt"},
                "tracker_config": {"max_age": 30}
            })
            if pipeline.initialize():
                print("Pipeline ready for processing")
            else:
                print("Pipeline initialization failed, check logs")
            ```
        """
        try:
            # Initialize detector
            self.detector = Detector(self.detector_config)
            if not self.detector.initialize():
                logger.error("Failed to initialize detector in pipeline")
                return False
            
            # Initialize tracker if enabled
            if self.enable_tracking:
                self.tracker = Tracker(self.tracker_config)
                if not self.tracker.initialize():
                    logger.error("Failed to initialize tracker in pipeline")
                    return False
            
            self.is_initialized = True
            logger.info("Pipeline initialized successfully")
            return True
        except (ValueError, RuntimeError) as e:
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing pipeline: {e}", exc_info=True)
            return False
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process image through detection and tracking pipeline
        
        This method processes an input image through the complete vision pipeline:
        1. Object detection using the configured detector
        2. Object tracking using the configured tracker (if enabled)
        
        If the pipeline is not initialized, it will attempt to initialize automatically.
        
        Args:
            image: Input image in BGR format (OpenCV standard), numpy array with shape (H, W, 3).
                   Data type should be uint8 with values in range [0, 255].
        
        Returns:
            Dict[str, Any]: Dictionary containing:
                - "detections": List[Detection] - List of detected objects from current frame
                - "tracks": List[Track] - List of tracked objects (if tracking enabled, else empty list)
                
                Each Detection contains:
                - bbox: Tuple of (x1, y1, x2, y2) coordinates
                - confidence: Confidence score
                - class_id: Integer class ID
                - class_name: String class name
                
                Each Track contains:
                - track_id: Unique integer identifier
                - bbox: Current bounding box
                - confidence: Current confidence
                - class_id: Object class ID
                - class_name: Object class name
                - age: Frames since track creation
                - time_since_update: Frames since last update
                - history: Previous positions (if available)
            
            Returns {"detections": [], "tracks": []} if:
                - Pipeline is not initialized and initialization fails
                - Detector is None
                - An error occurs during processing
        
        Raises:
            RuntimeError: If pipeline is not initialized and automatic initialization fails
            ValueError: If image is invalid (wrong format, shape, or data type)
        
        Example:
            ```python
            pipeline = VisionPipeline({
                "enable_tracking": True,
                "detector_config": {"model_path": "yolov8n.pt"}
            })
            pipeline.initialize()
            
            # Process each frame
            results = pipeline.process(frame)
            detections = results["detections"]
            tracks = results["tracks"]
            
            print(f"Detected {len(detections)} objects, tracking {len(tracks)} targets")
            ```
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("Pipeline not initialized and auto-initialization failed")
                return {"detections": [], "tracks": []}
        
        if self.detector is None:
            logger.error("Detector is None, cannot process")
            return {"detections": [], "tracks": []}
        
        results: Dict[str, Any] = {}
        
        try:
            # Run detection
            detections = self.detector.process(image)
            results["detections"] = detections
            
            # Run tracking if enabled
            if self.enable_tracking and self.tracker is not None:
                tracks = self.tracker.process(detections, image=image)
                results["tracks"] = tracks
            else:
                results["tracks"] = []
            
            return results
        except Exception as e:
            logger.error(f"Error during pipeline processing: {e}", exc_info=True)
            return {"detections": [], "tracks": []}
    
    def process_frame(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Alias for process method
        
        This method is provided for clarity when processing video frames.
        It is functionally equivalent to process().
        
        Args:
            image: Input image frame in BGR format (numpy array, shape: (H, W, 3))
        
        Returns:
            Dict[str, Any]: Dictionary containing detections and tracks
        """
        return self.process(image)
    
    def reset(self) -> None:
        """
        Reset pipeline state
        
        This method resets both the detector and tracker to their initial states,
        clearing all tracks and resetting internal state. The pipeline remains
        initialized but will start fresh.
        
        Note:
            The detector state is also reset, which may cause models to reload
            on the next process() call if lazy initialization is used.
        """
        super().reset()
        if self.tracker is not None:
            self.tracker.reset()
    
    def get_detector(self) -> Optional[Detector]:
        """
        Get detector instance
        
        Returns the internal detector instance, allowing direct access to
        detector methods and properties if needed.
        
        Returns:
            Optional[Detector]: Detector instance if initialized, None otherwise.
        
        Example:
            ```python
            pipeline = VisionPipeline()
            pipeline.initialize()
            
            detector = pipeline.get_detector()
            if detector:
                info = detector.get_model_info()
                print(f"Using model: {info['model_type']}")
            ```
        """
        return self.detector
    
    def get_tracker(self) -> Optional[Tracker]:
        """
        Get tracker instance
        
        Returns the internal tracker instance, allowing direct access to
        tracker methods and properties if needed.
        
        Returns:
            Optional[Tracker]: Tracker instance if initialized and tracking is enabled, None otherwise.
        
        Example:
            ```python
            pipeline = VisionPipeline({"enable_tracking": True})
            pipeline.initialize()
            
            tracker = pipeline.get_tracker()
            if tracker:
                active_tracks = tracker.get_tracks()
                print(f"Active tracks: {len(active_tracks)}")
            ```
        """
        return self.tracker

