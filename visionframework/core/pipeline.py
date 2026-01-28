"""
Pipeline for combining detection and tracking

This module provides a complete vision processing pipeline that combines
object detection and tracking in a single, easy-to-use interface.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from .base import BaseModule
from .detector import Detector
from .tracker import Tracker
from ..data.detection import Detection
from ..data.track import Track
from ..utils.monitoring.logger import get_logger
from ..utils.monitoring.performance import PerformanceMonitor
from ..utils.memory import create_memory_pool, acquire_memory, release_memory, optimize_memory_usage
from ..utils.concurrent import parallel_map

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
        Initialize vision pipeline with default configuration if none provided
        
        Args:
            config: Configuration dictionary with keys:
                - detector_config: Configuration dictionary for the detector
                - tracker_config: Configuration dictionary for the tracker
                - pose_estimator_config: Configuration dictionary for the pose estimator
                - enable_tracking: Boolean flag to enable/disable tracking (default: False)
                - enable_pose_estimation: Boolean flag to enable/disable pose estimation (default: False)
        
        Example:
            # Simplest usage - uses default YOLOv8n model
            pipeline = VisionPipeline()
            results = pipeline.process(frame)  # Auto-initializes if needed
            
            # With custom configuration
            pipeline = VisionPipeline({
                "detector_config": {"model_path": "yolov8s.pt", "conf_threshold": 0.3},
                "enable_tracking": True,
                "enable_pose_estimation": True
            })
        """
        # Initialize with minimal default config
        super().__init__({})
        
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
        
        # Update with user-provided config
        if config:
            # Update top-level config
            self.config.update(config)
            
            # Deep merge detector, tracker, and pose estimator configs
            if "detector_config" in config:
                self.config["detector_config"].update(config["detector_config"])
            if "tracker_config" in config:
                self.config["tracker_config"].update(config["tracker_config"])
            if "pose_estimator_config" in config:
                self.config["pose_estimator_config"].update(config["pose_estimator_config"])
        
        # Initialize core attributes
        self.detector: Optional[Detector] = None
        self.tracker: Optional[Tracker] = None
        self.pose_estimator: Optional[Any] = None
        self.enable_tracking: bool = self.config["enable_tracking"]
        self.enable_pose_estimation: bool = self.config["enable_pose_estimation"]
        self.enable_performance_monitoring: bool = self.config.get("enable_performance_monitoring", False)
        self.performance_metrics: List[str] = self.config.get("performance_metrics", [])
        self.detector_config: Dict[str, Any] = self.config["detector_config"]
        self.tracker_config: Dict[str, Any] = self.config["tracker_config"]
        self.pose_estimator_config: Dict[str, Any] = self.config["pose_estimator_config"]
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
    @classmethod
    def with_tracking(cls, config: Optional[Dict[str, Any]] = None) -> 'VisionPipeline':
        """
        Create a VisionPipeline with tracking enabled by default
        
        Args:
            config: Optional configuration dictionary
            
        Returns:
            VisionPipeline: Pipeline instance with tracking enabled
            
        Example:
            pipeline = VisionPipeline.with_tracking()
            results = pipeline.process(frame)  # Returns both detections and tracks
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
            enable_tracking: Whether to enable tracking (default: False)
            conf_threshold: Confidence threshold for detections (default: 0.25)
            
        Returns:
            VisionPipeline: Pipeline instance configured with the specified model
            
        Example:
            pipeline = VisionPipeline.from_model("yolov8m.pt", enable_tracking=True)
        """
        config = {
            "enable_tracking": enable_tracking,
            "detector_config": {
                "model_path": model_path,
                "conf_threshold": conf_threshold
            }
        }
        return cls(config)
    
    @staticmethod
    def process_image(image: np.ndarray, model_path: str = "yolov8n.pt", enable_tracking: bool = False, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Process a single image with minimal configuration
        
        This static method provides a convenient way to process a single image without manually
        creating and initializing a VisionPipeline instance.
        
        Args:
            image: Input image in BGR format (numpy array)
            model_path: Path to the detection model (default: "yolov8n.pt")
            enable_tracking: Whether to enable tracking (default: False)
            conf_threshold: Confidence threshold for detections (default: 0.25)
            
        Returns:
            Dict[str, Any]: Processing results containing detections and optionally tracks
            
        Example:
            ```python
            import cv2
            from visionframework import VisionPipeline
            
            # Load image
            image = cv2.imread("input.jpg")
            
            # Process with default model
            results = VisionPipeline.process_image(image)
            detections = results["detections"]
            
            # Process with custom model and tracking
            results = VisionPipeline.process_image(
                image, 
                model_path="yolov8m.pt",
                enable_tracking=True,
                conf_threshold=0.3
            )
            ```
        """
        pipeline = VisionPipeline.from_model(model_path, enable_tracking, conf_threshold)
        pipeline.initialize()
        return pipeline.process(image)
    
    @staticmethod
    def run_video(
        input_source: Union[str, int],
        output_path: Optional[str] = None,
        model_path: str = "yolov8n.pt",
        enable_tracking: bool = False,
        conf_threshold: float = 0.25,
        batch_size: int = 0,
        use_pyav: bool = False,
        **kwargs
    ) -> bool:
        """
        Run video processing with minimal configuration
        
        This static method provides a convenient way to process videos, camera streams, or network streams
        with just one line of code.
        
        Args:
            input_source: Path to video file, video stream URL, or camera index
            output_path: Optional path to save processed video
            model_path: Path to the detection model (default: "yolov8n.pt")
            enable_tracking: Whether to enable tracking (default: False)
            conf_threshold: Confidence threshold for detections (default: 0.25)
            batch_size: Batch size for processing (0 for non-batch processing, default: 0)
            use_pyav: Whether to use pyav for video processing (default: False, use OpenCV)
                      Note: PyAV currently only supports video files, not cameras or streams
            **kwargs: Additional arguments passed to process_video
            
        Returns:
            bool: True if processing completed successfully, False otherwise
        """
        pipeline = VisionPipeline.from_model(model_path, enable_tracking, conf_threshold)
        pipeline.initialize()
        
        # Enable batch inference if batch_size > 0
        if batch_size > 0 and hasattr(pipeline.detector, 'detector_impl'):
            pipeline.detector.detector_impl.batch_inference = True
        
        return pipeline.process_video(
            input_source=input_source,
            output_path=output_path,
            batch_size=batch_size if batch_size > 0 else None,
            use_pyav=use_pyav,
            **kwargs
        )
    
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
            - enable_performance_monitoring is a boolean (if provided)
            - performance_metrics is a list (if provided)
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
        
        # Validate performance monitoring settings
        if "enable_performance_monitoring" in config:
            enable_perf = config["enable_performance_monitoring"]
            if not isinstance(enable_perf, bool):
                return False, f"enable_performance_monitoring must be a boolean, got {type(enable_perf).__name__}"
        
        if "performance_metrics" in config:
            perf_metrics = config["performance_metrics"]
            if not isinstance(perf_metrics, list):
                return False, f"performance_metrics must be a list, got {type(perf_metrics).__name__}"
        
        # Validate pose estimation settings
        if "enable_pose_estimation" in config:
            enable_pose = config["enable_pose_estimation"]
            if not isinstance(enable_pose, bool):
                return False, f"enable_pose_estimation must be a boolean, got {type(enable_pose).__name__}"
        
        if "pose_estimator_config" in config:
            pose_config = config["pose_estimator_config"]
            if not isinstance(pose_config, dict):
                return False, f"pose_estimator_config must be a dictionary, got {type(pose_config).__name__}"
        
        return True, None
    
    def initialize(self) -> bool:
        """
        Initialize detector, tracker, and performance monitor
        
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
            3. Initialize performance monitor (if enable_performance_monitoring is True)
            4. Set is_initialized flag if all components initialized successfully
        
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
            logger.info("Initializing detector...")
            self.detector = Detector(self.detector_config)
            if not self.detector.initialize():
                logger.error("Failed to initialize detector in pipeline")
                return False
            
            # Initialize tracker if enabled
            if self.enable_tracking:
                logger.info("Initializing tracker...")
                self.tracker = Tracker(self.tracker_config)
                if not self.tracker.initialize():
                    logger.error("Failed to initialize tracker in pipeline")
                    return False
            
            # Initialize pose estimator if enabled
            if self.enable_pose_estimation:
                logger.info("Initializing pose estimator...")
                from .pose_estimator import PoseEstimator
                self.pose_estimator = PoseEstimator(self.pose_estimator_config)
                if not self.pose_estimator.initialize():
                    logger.error("Failed to initialize pose estimator in pipeline")
                    return False
            
            # Initialize performance monitor if enabled
            if self.enable_performance_monitoring:
                logger.info("Initializing performance monitor...")
                self.performance_monitor = PerformanceMonitor(metrics=self.performance_metrics)
            
            # Initialize memory pools
            self._initialize_memory_pools()
            
            self.is_initialized = True
            logger.info("Pipeline initialized successfully")
            return True
        except Exception as e:
            from ..exceptions import PipelineIntegrationError
            logger.error(f"Failed to initialize pipeline: {e}", exc_info=True)
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
                - "poses": List[Pose] - List of pose estimations (if pose estimation enabled, else empty list)
                
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
            
            Returns {"detections": [], "tracks": [], "poses": []} if:
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
            from ..exceptions import ProcessingError
            logger.error(f"Error during pipeline processing: {e}", exc_info=True)
            return {"detections": [], "tracks": [], "poses": []}
    
    def process_batch(self, images: List[np.ndarray], max_batch_size: Optional[int] = None, use_parallel: bool = False, max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process multiple images in a batch through detection and tracking pipeline
        
        This method processes multiple images efficiently by batching them through
        the detector when possible (if the detector supports batch inference).
        For tracking, each frame is processed sequentially to maintain temporal coherence.
        
        Args:
            images: List of input images in BGR format (OpenCV standard).
                   Each image should be numpy array with shape (H, W, 3) and uint8 data type.
            max_batch_size: Maximum batch size for processing. If None, use all images in one batch.
                           Useful for memory-constrained environments.
            use_parallel: Whether to use parallel processing for individual images
            max_workers: Maximum number of workers for parallel processing
        
        Returns:
            List[Dict[str, Any]]: List of result dictionaries, one per image.
                Each dictionary contains:
                - "detections": List[Detection] - Detected objects in that frame
                - "tracks": List[Track] - Tracked objects in that frame (if tracking enabled)
                - "poses": List[Pose] - Pose estimations in that frame (if pose estimation enabled)
                - "frame_idx": int - Index of the frame in the input list
        
        Example:
            ```python
            pipeline = VisionPipeline({
                "enable_tracking": True,
                "detector_config": {
                    "model_path": "yolov8n.pt",
                    "batch_inference": True  # Enable batch inference for better throughput
                }
            })
            pipeline.initialize()
            
            # Process multiple frames at once
            frames = [frame1, frame2, frame3, ...]
            results = pipeline.process_batch(frames)
            
            # Process with parallel processing
            results = pipeline.process_batch(frames, use_parallel=True, max_workers=4)
            
            for idx, result in enumerate(results):
                print(f"Frame {idx}: {len(result['detections'])} detections, {len(result['tracks'])} tracks")
            ```
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("Pipeline not initialized and auto-initialization failed")
                return [{"detections": [], "tracks": [], "poses": [], "frame_idx": i} for i in range(len(images))]
        
        if self.detector is None:
            logger.error("Detector is None, cannot process")
            return [{"detections": [], "tracks": [], "poses": [], "frame_idx": i} for i in range(len(images))]
        
        if not images:
            logger.warning("Empty image list provided to process_batch")
            return []
        
        try:
            # Split into smaller batches if max_batch_size is specified
            if max_batch_size and len(images) > max_batch_size:
                logger.debug(f"Splitting {len(images)} images into batches of {max_batch_size}")
                # Process in chunks
                chunk_results = []
                for i in range(0, len(images), max_batch_size):
                    chunk = images[i:i + max_batch_size]
                    chunk_result = self._process_batch_chunk(chunk, i, use_parallel, max_workers)
                    chunk_results.extend(chunk_result)
                return chunk_results
            else:
                # Process all images in one batch
                return self._process_batch_chunk(images, 0, use_parallel, max_workers)
        except Exception as e:
            from ..exceptions import BatchProcessingError
            logger.error(f"Error during batch pipeline processing: {e}", exc_info=True)
            return [{"detections": [], "tracks": [], "poses": [], "frame_idx": i} for i in range(len(images))]
    
    def _process_batch_chunk(self, images: List[np.ndarray], start_frame_idx: int = 0, use_parallel: bool = False, max_workers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process a chunk of images in a batch
        
        Args:
            images: List of input images in BGR format
            start_frame_idx: Starting frame index for this chunk
            use_parallel: Whether to use parallel processing
            max_workers: Maximum number of workers for parallel processing
            
        Returns:
            List of result dictionaries, one per image
        """
        results: List[Dict[str, Any]] = []
        
        # Check if detector supports batch processing via batch_inference flag
        detector_impl = getattr(self.detector, 'detector_impl', None)
        use_batch = getattr(detector_impl, 'batch_inference', False) if detector_impl else False
        
        if use_batch and len(images) > 1:
            # Use batch inference if supported and multiple images
            logger.debug(f"Using batch inference for {len(images)} images")
            all_detections = self.detector.process(images)
            
            # If detector returns detections for all images
            if isinstance(all_detections, list) and len(all_detections) > 0:
                # Check if first element is a list (batch) or Detection object
                if isinstance(all_detections[0], list):
                    # Batch results - one list per image
                    detections_per_image = all_detections
                else:
                    # Single image result or flat list - wrap in list
                    detections_per_image = []
                    for i in range(len(images)):
                        if i < len(all_detections):
                            det = all_detections[i]
                            detections_per_image.append([det] if not isinstance(det, list) else det)
                        else:
                            detections_per_image.append([])
            else:
                detections_per_image = [[] for _ in range(len(images))]
        else:
            # Process each image individually
            logger.debug(f"Processing {len(images)} images individually")
            
            if use_parallel and len(images) > 1:
                # Use parallel processing
                logger.debug(f"Using parallel processing with {max_workers or 'auto'} workers")
                
                def process_single_image(image):
                    return self.detector.process(image)
                
                detections_per_image = parallel_map(
                    images, 
                    process_single_image, 
                    max_workers=max_workers,
                    use_processes=False  # Use threads for I/O-bound operations
                )
            else:
                # Process sequentially
                detections_per_image = []
                for image in images:
                    dets = self.detector.process(image)
                    detections_per_image.append(dets)
        
        # Process tracking and pose estimation for each frame
        if use_parallel and len(images) > 1 and not self.enable_tracking:
            # Use parallel processing for post-processing (if no tracking)
            logger.debug(f"Using parallel post-processing")
            
            def process_post(image_and_detections):
                image, detections, frame_idx = image_and_detections
                frame_result: Dict[str, Any] = {
                    "detections": detections,
                    "frame_idx": start_frame_idx + frame_idx
                }
                
                # Run pose estimation if enabled
                if self.enable_pose_estimation and self.pose_estimator is not None:
                    poses = self.pose_estimator.process(image)
                    frame_result["poses"] = poses
                else:
                    frame_result["poses"] = []
                
                return frame_result
            
            # Prepare data for parallel processing
            data_for_parallel = [(img, dets, idx) for idx, (img, dets) in enumerate(zip(images, detections_per_image))]
            
            # Process in parallel
            parallel_results = parallel_map(
                data_for_parallel, 
                process_post, 
                max_workers=max_workers,
                use_processes=False
            )
            
            results.extend(parallel_results)
        else:
            # Process sequentially (required for tracking to maintain state)
            for frame_idx, (detections, image) in enumerate(zip(detections_per_image, images)):
                frame_result: Dict[str, Any] = {
                    "detections": detections,
                    "frame_idx": start_frame_idx + frame_idx
                }
                
                # Run tracking if enabled
                if self.enable_tracking and self.tracker is not None:
                    tracks = self.tracker.process(detections, image=image)
                    frame_result["tracks"] = tracks
                else:
                    frame_result["tracks"] = []
                
                # Run pose estimation if enabled
                if self.enable_pose_estimation and self.pose_estimator is not None:
                    poses = self.pose_estimator.process(image)
                    frame_result["poses"] = poses
                else:
                    frame_result["poses"] = []
                
                results.append(frame_result)
        
        return results
    
    def process_video_batch(
        self, 
        input_source: Union[str, int], 
        output_path: Optional[str] = None, 
        start_frame: int = 0, 
        end_frame: Optional[int] = None, 
        batch_size: int = 8,
        skip_frames: int = 0,
        frame_callback: Optional[Callable[[np.ndarray, int, Dict[str, Any]], np.ndarray]] = None,
        progress_callback: Optional[Callable[[float, int, int], None]] = None
    ) -> bool:
        """
        Process video file, camera stream, or video stream with batch processing for improved performance
        
        This method processes video frames in batches through the detector when supported,
        providing significant performance improvements for compatible detectors.
        
        Args:
            input_source: Path to video file, video stream URL, or camera index (0 for default webcam)
                          Supported stream formats: RTSP, HTTP, etc.
            output_path: Optional path to save processed video. If None, video is not saved.
            start_frame: Start frame number (default: 0)
            end_frame: End frame number (None for all frames, useful for video files only)
            batch_size: Number of frames to process in each batch (default: 8)
            skip_frames: Number of frames to skip between processing (default: 0)
            frame_callback: Optional callback function(frame, frame_number, results) -> processed_frame
                           Called after each frame is processed, can be used for visualization
            progress_callback: Optional callback function(progress, current_frame, total_frames) -> None
                              Called periodically with processing progress
        
        Returns:
            bool: True if processing completed successfully, False otherwise
        
        Example:
            ```python
            # Process video with batch processing
            pipeline = VisionPipeline.with_tracking({
                "detector_config": {
                    "model_path": "yolov8n.pt",
                    "batch_inference": True  # Enable batch inference for better performance
                }
            })
            pipeline.initialize()
            
            # Process RTSP stream with batch processing
            success = pipeline.process_video_batch(
                input_source="rtsp://example.com/stream",
                output_path="output_stream.mp4",
                batch_size=16,  # Process 16 frames in each batch
                progress_callback=lambda p, c, t: print(f"Progress: {p:.1%}")
            )
            ```
        """
        from ..utils.io.video_utils import VideoProcessor, VideoWriter, PyAVVideoProcessor, PyAVVideoWriter
        
        # Try to import pyav for optional usage
        try:
            import av
            pyav_available = True
        except ImportError:
            pyav_available = False
        
        # Determine if we can use pyav
        use_pyav_final = use_pyav and pyav_available
        
        # Check if input is a video file (PyAV doesn't support cameras)
        if use_pyav_final:
            is_camera = isinstance(input_source, int) or (isinstance(input_source, str) and input_source == "0")
            is_stream = isinstance(input_source, str) and (input_source.startswith("rtsp://") or \
                                                         input_source.startswith("http://") or \
                                                         input_source.startswith("https://"))
            if is_camera or is_stream:
                logger.info("PyAV only supports video files, not cameras or streams. Falling back to OpenCV.")
                use_pyav_final = False
        
        # Initialize video processor
        if use_pyav_final:
            processor = PyAVVideoProcessor(input_source)
        else:
            processor = VideoProcessor(input_source)
        
        if not processor.open():
            logger.error(f"Failed to open video source: {input_source}")
            return False
        
        # Get video info
        video_info = processor.get_info()
        total_frames = video_info["total_frames"]
        if end_frame is not None:
            total_frames = min(total_frames, end_frame)
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            if use_pyav_final:
                writer = PyAVVideoWriter(
                    output_path,
                    fps=video_info["fps"],
                    frame_size=(video_info["width"], video_info["height"])
                )
            else:
                writer = VideoWriter(
                    output_path,
                    fps=video_info["fps"],
                    frame_size=(video_info["width"], video_info["height"])
                )
            if not writer.open():
                logger.error(f"Failed to open video writer: {output_path}")
                return False
        
        try:
            frame_count = 0
            processed_count = 0
            batch_frames = []
            batch_frame_nums = []
            
            while True:
                ret, frame = processor.read_frame()
                if not ret:
                    # Process remaining frames in batch
                    if batch_frames:
                        # Process batch
                        batch_results = self.process_batch(batch_frames)
                        
                        # Handle each frame in batch
                        for batch_idx, (frame_to_process, frame_num, result) in enumerate(zip(batch_frames, batch_frame_nums, batch_results)):
                            # Apply frame callback if provided
                            processed_frame = frame_to_process.copy()
                            if frame_callback:
                                processed_frame = frame_callback(processed_frame, frame_num, result)
                            
                            # Write frame to output video if writer is available
                            if writer:
                                writer.write(processed_frame)
                        
                        batch_frames.clear()
                        batch_frame_nums.clear()
                    break
                
                current_frame = processor.current_frame_num
                
                # Check if we've reached the end frame
                if end_frame is not None and current_frame > end_frame:
                    break
                
                # Skip frames before start_frame
                if current_frame < start_frame:
                    continue
                
                # Skip frames based on skip_frames parameter
                if (current_frame - start_frame) % (skip_frames + 1) != 0:
                    continue
                
                # Add frame to batch
                batch_frames.append(frame)
                batch_frame_nums.append(current_frame)
                
                # Process batch when it reaches batch_size
                if len(batch_frames) >= batch_size:
                    # Process batch
                    batch_results = self.process_batch(batch_frames)
                    
                    # Handle each frame in batch
                    for batch_idx, (frame_to_process, frame_num, result) in enumerate(zip(batch_frames, batch_frame_nums, batch_results)):
                        # Apply frame callback if provided
                        processed_frame = frame_to_process.copy()
                        if frame_callback:
                            processed_frame = frame_callback(processed_frame, frame_num, result)
                        
                        # Write frame to output video if writer is available
                        if writer:
                            writer.write(processed_frame)
                    
                    processed_count += len(batch_frames)
                    frame_count += len(batch_frames)
                    
                    # Update progress if callback is provided
                    if progress_callback and total_frames > 0:
                        progress = min(1.0, (current_frame - start_frame) / (total_frames - start_frame))
                        progress_callback(progress, current_frame, total_frames)
                    
                    # Clear batch for next iteration
                    batch_frames.clear()
                    batch_frame_nums.clear()
            
            logger.info(f"Video processing completed. Processed {processed_count} frames out of {frame_count} total frames.")
            return True
        except Exception as e:
            logger.error(f"Error during batch video processing: {e}", exc_info=True)
            return False
        finally:
            # Cleanup resources
            processor.close()
            if writer:
                writer.close()
    
    def reset(self) -> None:
        """
        Reset pipeline state
        
        This method resets all components (detector, tracker, pose estimator) to their initial states,
        clearing all tracks, poses, and resetting internal state. The pipeline remains
        initialized but will start fresh.
        
        Note:
            The detector state is also reset, which may cause models to reload
            on the next process() call if lazy initialization is used.
        """
        super().reset()
        if self.tracker is not None:
            self.tracker.reset()
        # Pose estimator doesn't have a reset method yet, but we can add one if needed in the future
    
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
            Optional[Tracker]: Tracker instance if initialized, None otherwise.
        
        Example:
            ```python
            pipeline = VisionPipeline()
            pipeline.initialize()
            
            tracker = pipeline.get_tracker()
            if tracker:
                active_tracks = tracker.get_tracks()
                print(f"Active tracks: {len(active_tracks)}")
            ```
        """
        return self.tracker
    
    def get_pose_estimator(self) -> Optional[Any]:
        """
        Get pose estimator instance
        
        Returns the internal pose estimator instance, allowing direct access to
        pose estimator methods and properties if needed.
        
        Returns:
            Optional[Any]: Pose estimator instance if initialized, None otherwise.
        
        Example:
            ```python
            pipeline = VisionPipeline()
            pipeline.initialize()
            
            pose_estimator = pipeline.get_pose_estimator()
            if pose_estimator:
                info = pose_estimator.get_model_info()
                print(f"Using pose model: {info['model_path']}")
            ```
        """
        return self.pose_estimator
    
    def _initialize_memory_pools(self) -> None:
        """
        Initialize memory pools for efficient memory allocation and reuse
        """
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources held by pipeline components"""
        try:
            if self.detector is not None:
                self.detector.cleanup()
            if self.tracker is not None:
                self.tracker.cleanup()
            if self.pose_estimator is not None:
                # Pose estimator doesn't have a cleanup method yet, but we can add one if needed in the future
                pass
            # Optimize memory usage by clearing unused memory
            optimize_memory_usage()
        finally:
            self.detector = None
            self.tracker = None
            self.pose_estimator = None
            self.is_initialized = False
    
    def shutdown(self) -> None:
        """Shutdown pipeline and cleanup resources"""
        self.cleanup()
    
    def process_video(
        self, 
        input_source: Union[str, int], 
        output_path: Optional[str] = None, 
        start_frame: int = 0, 
        end_frame: Optional[int] = None, 
        skip_frames: int = 0,
        frame_callback: Optional[Callable[[np.ndarray, int, Dict[str, Any]], np.ndarray]] = None,
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
        use_pyav: bool = False
    ) -> bool:
        """
        Process video file, camera stream, or video stream (RTSP/HTTP) through the vision pipeline
        
        This method provides a convenient way to process entire video files, camera streams, or network streams
        without manually writing frame loops. It handles video reading, processing, and writing
        in a single method call.
        
        Args:
            input_source: Path to video file, video stream URL, or camera index (0 for default webcam)
                          Supported stream formats: RTSP, HTTP, etc.
            output_path: Optional path to save processed video. If None, video is not saved.
            start_frame: Start frame number (default: 0)
            end_frame: End frame number (None for all frames, useful for video files only)
            skip_frames: Number of frames to skip between processing (default: 0)
            frame_callback: Optional callback function(frame, frame_number, results) -> processed_frame
                           Called after each frame is processed, can be used for visualization
            progress_callback: Optional callback function(progress, current_frame, total_frames) -> None
                              Called periodically with processing progress
            use_pyav: Whether to use pyav for video processing (default: False, use OpenCV)
                      Note: PyAV currently only supports video files, not cameras or streams
        
        Returns:
            bool: True if processing completed successfully, False otherwise
        
        Example:
            ```python
            # Process video file and save results
            pipeline = VisionPipeline.with_tracking()
            pipeline.initialize()
            
            # Process video with progress callback
            def progress(progress_val, current, total):
                print(f"Progress: {progress_val:.1%} ({current}/{total})")
            
            # Process video file
            success = pipeline.process_video(
                input_source="input_video.mp4",
                output_path="output_video.mp4",
                start_frame=0,
                end_frame=1000,
                skip_frames=0,
                progress_callback=progress
            )
            
            # Process RTSP stream
            success = pipeline.process_video(
                input_source="rtsp://username:password@example.com/stream",
                output_path="output_stream.mp4",
                frame_callback=lambda frame, fn, res: frame  # No visualization
            )
            
            # Process camera stream
            success = pipeline.process_video(
                input_source=0,  # Default webcam
                output_path="output_camera.mp4"
            )
            ```
        """
        from ..utils.io.video_utils import VideoProcessor, VideoWriter
        
        # Initialize video processor
        processor = VideoProcessor(input_source)
        if not processor.open():
            logger.error(f"Failed to open video source: {input_source}")
            return False
        
        # Get video info
        video_info = processor.get_info()
        total_frames = video_info["total_frames"]
        if end_frame is not None:
            total_frames = min(total_frames, end_frame)
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            writer = VideoWriter(
                output_path,
                fps=video_info["fps"],
                frame_size=(video_info["width"], video_info["height"])
            )
            if not writer.open():
                logger.error(f"Failed to open video writer: {output_path}")
                return False
        
        try:
            frame_count = 0
            processed_count = 0
            
            while True:
                ret, frame = processor.read_frame()
                if not ret:
                    break
                
                current_frame = processor.current_frame_num
                
                # Check if we've reached the end frame
                if end_frame is not None and current_frame > end_frame:
                    break
                
                # Skip frames before start_frame
                if current_frame < start_frame:
                    continue
                
                # Skip frames based on skip_frames parameter
                if (current_frame - start_frame) % (skip_frames + 1) != 0:
                    continue
                
                # Process frame using the pipeline
                results = self.process(frame)
                
                # Apply frame callback if provided
                processed_frame = frame.copy()
                if frame_callback:
                    processed_frame = frame_callback(processed_frame, current_frame, results)
                
                # Write frame to output video if writer is available
                if writer:
                    writer.write(processed_frame)
                
                processed_count += 1
                frame_count += 1
                
                # Update progress if callback is provided
                if progress_callback and total_frames > 0:
                    progress = min(1.0, (current_frame - start_frame) / (total_frames - start_frame))
                    progress_callback(progress, current_frame, total_frames)
            
            logger.info(f"Video processing completed. Processed {processed_count} frames out of {frame_count} total frames.")
            return True
        except Exception as e:
            logger.error(f"Error during video processing: {e}", exc_info=True)
            return False
        finally:
            # Cleanup resources
            processor.close()
            if writer:
                writer.close()

