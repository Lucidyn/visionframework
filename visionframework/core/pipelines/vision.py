"""
Vision pipeline class for Vision Framework
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Union, Callable
from .base import BasePipeline
from ..components.detectors import YOLODetector
from ..components.trackers import ByteTracker
from ...data.detection import Detection
from ...data.track import Track
from ...data.pose import Pose
from ...utils.monitoring.logger import get_logger

logger = get_logger(__name__)


class VisionPipeline(BasePipeline):
    """
    Vision pipeline for object detection and tracking
    
    This class provides a unified interface for processing images or video frames
    through both detection and tracking stages.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize vision pipeline
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Set default config values
        self.config.setdefault("enable_tracking", False)
        self.config.setdefault("enable_pose_estimation", False)
        self.config.setdefault("detector_config", {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.25
        })
        self.config.setdefault("tracker_config", {
            "tracker_type": "bytetrack",
            "max_age": 30
        })
        self.config.setdefault("pose_estimator_config", {
            "model_path": "yolov8n-pose.pt",
            "conf_threshold": 0.25,
            "keypoint_threshold": 0.5
        })
        
        # Initialize attributes
        self.detector: Optional[Any] = None
        self.tracker: Optional[Any] = None
        self.pose_estimator: Optional[Any] = None
        self.enable_tracking: bool = self.config["enable_tracking"]
        self.enable_pose_estimation: bool = self.config["enable_pose_estimation"]
        self.detector_config: Dict[str, Any] = self.config["detector_config"]
        self.tracker_config: Dict[str, Any] = self.config["tracker_config"]
        self.pose_estimator_config: Dict[str, Any] = self.config["pose_estimator_config"]
    
    @classmethod
    def with_tracking(cls, config: Optional[Dict[str, Any]] = None) -> 'VisionPipeline':
        """
        Create a VisionPipeline with tracking enabled
        
        Args:
            config: Optional configuration dictionary
        
        Returns:
            VisionPipeline: Pipeline instance with tracking enabled
        """
        if config is None:
            config = {}
        config["enable_tracking"] = True
        return cls(config)
    
    @classmethod
    def from_model(cls, model_path: str, enable_tracking: bool = False, conf_threshold: float = 0.25) -> 'VisionPipeline':
        """
        Create a VisionPipeline from a specific model path
        
        Args:
            model_path: Path to the detection model
            enable_tracking: Whether to enable tracking
            conf_threshold: Confidence threshold for detections
        
        Returns:
            VisionPipeline: Pipeline instance configured with the specified model
        """
        config = {
            "enable_tracking": enable_tracking,
            "detector_config": {
                "model_path": model_path,
                "conf_threshold": conf_threshold
            }
        }
        return cls(config)
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate pipeline configuration
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Tuple[bool, Optional[str]]: (valid, error_message)
        """
        # Validate base config
        base_valid, base_error = super().validate_config(config)
        if not base_valid:
            return False, base_error
        
        # Validate enable_tracking
        if "enable_tracking" in config:
            enable_tracking = config["enable_tracking"]
            if not isinstance(enable_tracking, bool):
                return False, f"enable_tracking must be a boolean, got {type(enable_tracking).__name__}"
        
        # Validate enable_pose_estimation
        if "enable_pose_estimation" in config:
            enable_pose = config["enable_pose_estimation"]
            if not isinstance(enable_pose, bool):
                return False, f"enable_pose_estimation must be a boolean, got {type(enable_pose).__name__}"
        
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
        
        # Validate pose_estimator_config
        if "pose_estimator_config" in config:
            pose_config = config["pose_estimator_config"]
            if not isinstance(pose_config, dict):
                return False, f"pose_estimator_config must be a dictionary, got {type(pose_config).__name__}"
        
        return True, None
    
    def initialize(self) -> bool:
        """
        Initialize pipeline components

        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize base pipeline
            if not super().initialize():
                return False
            
            # Initialize detector
            logger.info("Initializing detector...")
            from ..components.detectors.yolo_detector import YOLODetector
            self.detector = YOLODetector(self.detector_config)
            if not hasattr(self.detector, 'initialize') or not self.detector.initialize():
                logger.error("Failed to initialize detector")
                return False
            
            # Initialize tracker if enabled
            if self.enable_tracking:
                logger.info("Initializing tracker...")
                from ..components.trackers.byte_tracker import ByteTracker
                self.tracker = ByteTracker(self.tracker_config)
                if not hasattr(self.tracker, 'initialize') or not self.tracker.initialize():
                    logger.error("Failed to initialize tracker")
                    return False
            
            # Initialize pose estimator if enabled
            if self.enable_pose_estimation:
                logger.info("Initializing pose estimator...")
                from ..components.processors.pose_estimator import PoseEstimator
                self.pose_estimator = PoseEstimator(self.pose_estimator_config)
                if not hasattr(self.pose_estimator, 'initialize') or not self.pose_estimator.initialize():
                    logger.error("Failed to initialize pose estimator")
                    return False
            
            logger.info("Vision pipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize vision pipeline: {e}")
            return False
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process a single image
        
        Args:
            image: Input image
        
        Returns:
            Dict[str, Any]: Processing results
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("Pipeline not initialized and auto-initialization failed")
                return {"detections": [], "tracks": [], "poses": []}
        
        if self.detector is None:
            logger.error("Detector is None, cannot process")
            return {"detections": [], "tracks": [], "poses": []}
        
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
            
            # Run pose estimation if enabled
            if self.enable_pose_estimation and self.pose_estimator is not None:
                poses = self.pose_estimator.process(image)
                results["poses"] = poses
            else:
                results["poses"] = []
            
            return results
        except Exception as e:
            logger.error(f"Error during pipeline processing: {e}")
            return {"detections": [], "tracks": [], "poses": []}
    
    def process_batch(self, images: List[np.ndarray], max_batch_size: Optional[int] = None, use_parallel: bool = False, max_workers: Optional[int] = None, enable_memory_optimization: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch
        
        Args:
            images: List of input images
            max_batch_size: Maximum batch size for processing. If None, use all images in one batch.
            use_parallel: Whether to use parallel processing for individual images
            max_workers: Maximum number of workers for parallel processing
            enable_memory_optimization: Whether to enable memory optimization techniques
        
        Returns:
            List[Dict[str, Any]]: List of processing results
        """
        from .batch import BatchPipeline
        batch_pipeline = BatchPipeline(self.config)
        batch_pipeline.initialize()
        return batch_pipeline.process_batch(images, max_batch_size=max_batch_size, use_parallel=use_parallel, max_workers=max_workers, enable_memory_optimization=enable_memory_optimization)
    
    def process_video(self, input_source: Any, output_path: Optional[str] = None) -> bool:
        """
        Process video
        
        Args:
            input_source: Video source
            output_path: Output video path
        
        Returns:
            bool: True if processing successful
        """
        from .video import VideoPipeline
        video_pipeline = VideoPipeline(self.config)
        video_pipeline.initialize()
        return video_pipeline.process_video(input_source, output_path)
    
    def process_video_batch(
        self, 
        input_source: Union[str, int], 
        output_path: Optional[str] = None, 
        start_frame: int = 0, 
        end_frame: Optional[int] = None, 
        batch_size: int = 8,
        skip_frames: int = 0,
        frame_callback: Optional[Callable[[np.ndarray, int, Dict[str, Any]], np.ndarray]] = None,
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
        use_pyav: bool = False
    ) -> bool:
        """
        Process video file, camera stream, or video stream with batch processing for improved performance
        
        Args:
            input_source: Path to video file, video stream URL, or camera index (0 for default webcam)
            output_path: Optional path to save processed video. If None, video is not saved.
            start_frame: Start frame number (default: 0)
            end_frame: End frame number (None for all frames, useful for video files only)
            batch_size: Number of frames to process in each batch (default: 8)
            skip_frames: Number of frames to skip between processing (default: 0)
            frame_callback: Optional callback function(frame, frame_number, results) -> processed_frame
            progress_callback: Optional callback function(progress, current_frame, total_frames) -> None
            use_pyav: Whether to use PyAV for video processing (default: False)
        
        Returns:
            bool: True if processing completed successfully, False otherwise
        """
        from .video import VideoPipeline
        video_pipeline = VideoPipeline(self.config)
        video_pipeline.initialize()
        # VideoPipeline.process_video supports batch_size parameter
        return video_pipeline.process_video(
            input_source, 
            output_path, 
            batch_size=batch_size,
            use_pyav=use_pyav,
            start_frame=start_frame,
            end_frame=end_frame,
            skip_frames=skip_frames,
            frame_callback=frame_callback,
            progress_callback=progress_callback
        )
    
    def reset(self) -> None:
        """
        Reset pipeline state
        """
        super().reset()
        if self.tracker is not None:
            self.tracker.reset()
        # Reset other components if needed
    
    def get_detector(self) -> Optional[Any]:
        """
        Get detector instance

        Returns:
            Optional[Any]: Detector instance if initialized
        """
        return self.detector
    
    def get_tracker(self) -> Optional[Any]:
        """
        Get tracker instance

        Returns:
            Optional[Any]: Tracker instance if initialized
        """
        return self.tracker
    
    def get_pose_estimator(self) -> Optional[Any]:
        """
        Get pose estimator instance
        
        Returns:
            Optional[Any]: Pose estimator instance if initialized
        """
        return self.pose_estimator
