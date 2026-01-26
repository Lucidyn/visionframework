"""
Pipeline for combining detection and tracking

This module provides a complete vision processing pipeline that combines
object detection and tracking in a single, easy-to-use interface.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from .base import BaseModule
from .detector import Detector
from .tracker import Tracker
from ..data.detection import Detection
from ..data.track import Track
from ..utils.monitoring.logger import get_logger

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
                - enable_tracking: Boolean flag to enable/disable tracking (default: False)
        
        Example:
            # Simplest usage - uses default YOLOv8n model
            pipeline = VisionPipeline()
            results = pipeline.process(frame)  # Auto-initializes if needed
            
            # With custom configuration
            pipeline = VisionPipeline({
                "detector_config": {"model_path": "yolov8s.pt", "conf_threshold": 0.3}
            })
        """
        # Initialize with minimal default config
        super().__init__({})
        
        # Set default config values
        self.config.setdefault("enable_tracking", False)
        self.config.setdefault("detector_config", {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.25
        })
        self.config.setdefault("tracker_config", {
            "tracker_type": "bytetrack",
            "max_age": 30
        })
        
        # Update with user-provided config
        if config:
            # Update top-level config
            self.config.update(config)
            
            # Deep merge detector and tracker configs
            if "detector_config" in config:
                self.config["detector_config"].update(config["detector_config"])
            if "tracker_config" in config:
                self.config["tracker_config"].update(config["tracker_config"])
        
        # Initialize core attributes
        self.detector: Optional[Detector] = None
        self.tracker: Optional[Tracker] = None
        self.enable_tracking: bool = self.config["enable_tracking"]
        self.detector_config: Dict[str, Any] = self.config["detector_config"]
        self.tracker_config: Dict[str, Any] = self.config["tracker_config"]
        
    @classmethod
    def with_tracking(cls, config: Optional[Dict[str, Any]] = None):
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
    def from_model(cls, model_path: str, enable_tracking: bool = False, conf_threshold: float = 0.25):
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
        input_source: str or int,
        output_path: Optional[str] = None,
        model_path: str = "yolov8n.pt",
        enable_tracking: bool = False,
        conf_threshold: float = 0.25,
        batch_size: int = 0,
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
            **kwargs: Additional arguments passed to process_video or process_video_batch
            
        Returns:
            bool: True if processing completed successfully, False otherwise
            
        Example:
            ```python
            from visionframework import VisionPipeline
            
            # Process video file with default settings
            VisionPipeline.run_video(
                input_source="input.mp4",
                output_path="output.mp4"
            )
            
            # Process RTSP stream with custom settings
            VisionPipeline.run_video(
                input_source="rtsp://example.com/stream",
                output_path="output_stream.mp4",
                model_path="yolov8s.pt",
                enable_tracking=True,
                conf_threshold=0.3,
                batch_size=16
            )
            ```
        """
        pipeline = VisionPipeline.from_model(model_path, enable_tracking, conf_threshold)
        pipeline.initialize()
        
        # Enable batch inference if batch_size > 0
        if batch_size > 0 and hasattr(pipeline.detector, 'detector_impl'):
            pipeline.detector.detector_impl.batch_inference = True
        
        if batch_size > 0:
            return pipeline.process_video_batch(
                input_source=input_source,
                output_path=output_path,
                batch_size=batch_size,
                **kwargs
            )
        else:
            return pipeline.process_video(
                input_source=input_source,
                output_path=output_path,
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
            
            # Initialize performance monitor if enabled
            if self.enable_performance_monitoring:
                logger.info("Initializing performance monitor...")
                self.performance_monitor = PerformanceMonitor(metrics=self.performance_metrics)
            
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
    
    def process_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Process multiple images in a batch through detection and tracking pipeline
        
        This method processes multiple images efficiently by batching them through
        the detector when possible (if the detector supports batch inference).
        For tracking, each frame is processed sequentially to maintain temporal coherence.
        
        Args:
            images: List of input images in BGR format (OpenCV standard).
                   Each image should be numpy array with shape (H, W, 3) and uint8 data type.
        
        Returns:
            List[Dict[str, Any]]: List of result dictionaries, one per image.
                Each dictionary contains:
                - "detections": List[Detection] - Detected objects in that frame
                - "tracks": List[Track] - Tracked objects in that frame (if tracking enabled)
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
            
            for idx, result in enumerate(results):
                print(f"Frame {idx}: {len(result['detections'])} detections, {len(result['tracks'])} tracks")
            ```
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("Pipeline not initialized and auto-initialization failed")
                return [{"detections": [], "tracks": [], "frame_idx": i} for i in range(len(images))]
        
        if self.detector is None:
            logger.error("Detector is None, cannot process")
            return [{"detections": [], "tracks": [], "frame_idx": i} for i in range(len(images))]
        
        if not images:
            logger.warning("Empty image list provided to process_batch")
            return []
        
        results: List[Dict[str, Any]] = []
        
        try:
            # Check if detector supports batch processing via batch_inference flag
            detector_impl = self.detector.detector_impl
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
                        detections_per_image = [[det] if not isinstance(det, list) else det 
                                               for det in all_detections[:len(images)]]
                else:
                    detections_per_image = [[] for _ in range(len(images))]
            else:
                # Process each image individually
                logger.debug(f"Processing {len(images)} images individually")
                detections_per_image = []
                for image in images:
                    dets = self.detector.process(image)
                    detections_per_image.append(dets)
            
            # Process tracking for each frame
            for frame_idx, detections in enumerate(detections_per_image):
                frame_result: Dict[str, Any] = {
                    "detections": detections,
                    "frame_idx": frame_idx
                }
                
                # Run tracking if enabled
                if self.enable_tracking and self.tracker is not None:
                    tracks = self.tracker.process(detections, image=images[frame_idx])
                    frame_result["tracks"] = tracks
                else:
                    frame_result["tracks"] = []
                
                results.append(frame_result)
            
            return results
        except Exception as e:
            logger.error(f"Error during batch pipeline processing: {e}", exc_info=True)
            return [{"detections": [], "tracks": [], "frame_idx": i} for i in range(len(images))]
    
    def process_video_batch(
        self, 
        input_source: str or int, 
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
    
    def process_video(
        self, 
        input_source: str or int, 
        output_path: Optional[str] = None, 
        start_frame: int = 0, 
        end_frame: Optional[int] = None, 
        skip_frames: int = 0,
        frame_callback: Optional[Callable[[np.ndarray, int, Dict[str, Any]], np.ndarray]] = None,
        progress_callback: Optional[Callable[[float, int, int], None]] = None
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
                results = self.process_frame(frame)
                
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

