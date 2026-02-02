"""
Video pipeline class for Vision Framework
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Union, Callable
from .base import BasePipeline
# Detector and Tracker imports are now handled via components
from ...data.detection import Detection
from ...data.track import Track
from ...data.pose import Pose
from ...utils.monitoring.logger import get_logger
from ...utils.io.video_utils import VideoProcessor, VideoWriter, PyAVVideoProcessor, PyAVVideoWriter

logger = get_logger(__name__)


class VideoPipeline(BasePipeline):
    """
    Video pipeline for processing video files, camera streams, or network streams
    
    This class provides optimized video processing capabilities, including
    batch processing and PyAV integration for improved performance.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize video pipeline
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Set default config values
        self.config.setdefault("detector_config", {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.25
        })
        self.config.setdefault("enable_tracking", False)
        self.config.setdefault("tracker_config", {
            "tracker_type": "bytetrack",
            "max_age": 30
        })
        self.config.setdefault("enable_pose_estimation", False)
        self.config.setdefault("pose_estimator_config", {
            "model_path": "yolov8n-pose.pt",
            "conf_threshold": 0.25
        })
        
        # Initialize attributes
        self.detector: Optional[Any] = None
        self.tracker: Optional[Any] = None
        self.pose_estimator: Optional[Any] = None
        self.enable_tracking: bool = self.config["enable_tracking"]
        self.enable_pose_estimation: bool = self.config["enable_pose_estimation"]
    
    def initialize(self) -> bool:
        """
        Initialize video pipeline components
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize base pipeline
            if not super().initialize():
                return False
            
            # Initialize detector
            logger.info("Initializing detector for video processing...")
            from ..components.detectors.yolo_detector import YOLODetector
            self.detector = YOLODetector(self.config["detector_config"])
            if not hasattr(self.detector, 'initialize') or not self.detector.initialize():
                logger.error("Failed to initialize detector")
                return False
            
            # Initialize tracker if enabled
            if self.enable_tracking:
                logger.info("Initializing tracker...")
                from ..components.trackers.byte_tracker import ByteTracker
                self.tracker = ByteTracker(self.config["tracker_config"])
                if not hasattr(self.tracker, 'initialize') or not self.tracker.initialize():
                    logger.error("Failed to initialize tracker")
                    return False
            
            # Initialize pose estimator if enabled
            if self.enable_pose_estimation:
                logger.info("Initializing pose estimator...")
                from ..components.processors.pose_estimator import PoseEstimator
                self.pose_estimator = PoseEstimator(self.config["pose_estimator_config"])
                if not hasattr(self.pose_estimator, 'initialize') or not self.pose_estimator.initialize():
                    logger.error("Failed to initialize pose estimator")
                    return False
            
            logger.info("Video pipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize video pipeline: {e}")
            return False
    
    def process_video(self, input_source: Union[str, int], output_path: Optional[str] = None, batch_size: int = 0, use_pyav: bool = False, start_frame: int = 0, end_frame: Optional[int] = None, skip_frames: int = 0, frame_callback: Optional[Callable] = None, progress_callback: Optional[Callable] = None) -> bool:
        """
        Process video file, camera stream, or network stream
        
        Args:
            input_source: Path to video file, video stream URL, or camera index
            output_path: Optional path to save processed video
            batch_size: Number of frames to process in each batch (0 for non-batch processing)
            use_pyav: Whether to use pyav for video processing
            start_frame: Start frame number
            end_frame: End frame number
            skip_frames: Number of frames to skip between processing
            frame_callback: Optional callback function for each processed frame
            progress_callback: Optional callback function for processing progress
        
        Returns:
            bool: True if processing completed successfully
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("Pipeline not initialized and auto-initialization failed")
                return False
        
        if self.detector is None:
            logger.error("Detector is None, cannot process video")
            return False
        
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
            if is_camera:
                logger.info("PyAV only supports video files and streams, not cameras. Falling back to OpenCV.")
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
            max_batch_size = batch_size if batch_size > 0 else 8
            
            while True:
                ret, frame = processor.read_frame()
                if not ret:
                    # Process remaining frames in batch
                    if batch_frames:
                        logger.debug(f"Processing final batch of {len(batch_frames)} frames")
                        try:
                            # Process batch
                            batch_results = self._process_video_batch(batch_frames)
                            
                            # Handle each frame in batch
                            for batch_idx, (frame_to_process, frame_num, result) in enumerate(zip(batch_frames, batch_frame_nums, batch_results)):
                                # Apply frame callback if provided
                                processed_frame = frame_to_process.copy()
                                if frame_callback:
                                    try:
                                        processed_frame = frame_callback(processed_frame, frame_num, result)
                                    except Exception as callback_error:
                                        logger.error(f"Error in frame callback: {callback_error}")
                                
                                # Write frame to output video if writer is available
                                if writer:
                                    try:
                                        writer.write(processed_frame)
                                    except Exception as write_error:
                                        logger.error(f"Error writing frame: {write_error}")
                            
                            processed_count += len(batch_frames)
                            frame_count += len(batch_frames)
                            
                            # Update final progress
                            if progress_callback and total_frames > 0:
                                progress = min(1.0, processed_count / (total_frames - start_frame + 1))
                                progress_callback(progress, processed_count, total_frames - start_frame + 1)
                            
                        except Exception as batch_error:
                            logger.error(f"Error processing final batch: {batch_error}")
                        finally:
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
                if batch_size > 0 and len(batch_frames) >= max_batch_size:
                    logger.debug(f"Processing batch of {len(batch_frames)} frames")
                    try:
                        # Process batch
                        batch_results = self._process_video_batch(batch_frames)
                        
                        # Handle each frame in batch
                        for batch_idx, (frame_to_process, frame_num, result) in enumerate(zip(batch_frames, batch_frame_nums, batch_results)):
                            # Apply frame callback if provided
                            processed_frame = frame_to_process.copy()
                            if frame_callback:
                                try:
                                    processed_frame = frame_callback(processed_frame, frame_num, result)
                                except Exception as callback_error:
                                    logger.error(f"Error in frame callback: {callback_error}")
                            
                            # Write frame to output video if writer is available
                            if writer:
                                try:
                                    writer.write(processed_frame)
                                except Exception as write_error:
                                    logger.error(f"Error writing frame: {write_error}")
                        
                        processed_count += len(batch_frames)
                        frame_count += len(batch_frames)
                        
                        # Update progress if callback is provided
                        if progress_callback and total_frames > 0:
                            progress = min(1.0, processed_count / (total_frames - start_frame + 1))
                            progress_callback(progress, processed_count, total_frames - start_frame + 1)
                        
                    except Exception as batch_error:
                        logger.error(f"Error processing batch: {batch_error}")
                    finally:
                        # Clear batch for next iteration
                        batch_frames.clear()
                        batch_frame_nums.clear()
                elif batch_size == 0:
                    # Process single frame immediately if batch processing is disabled
                    try:
                        # Process frame
                        result = self._process_video_frame(frame)
                        
                        # Apply frame callback if provided
                        processed_frame = frame.copy()
                        if frame_callback:
                            try:
                                processed_frame = frame_callback(processed_frame, current_frame, result)
                            except Exception as callback_error:
                                logger.error(f"Error in frame callback: {callback_error}")
                        
                        # Write frame to output video if writer is available
                        if writer:
                            try:
                                writer.write(processed_frame)
                            except Exception as write_error:
                                logger.error(f"Error writing frame: {write_error}")
                        
                        processed_count += 1
                        frame_count += 1
                        
                        # Update progress if callback is provided
                        if progress_callback and total_frames > 0:
                            progress = min(1.0, processed_count / (total_frames - start_frame + 1))
                            progress_callback(progress, processed_count, total_frames - start_frame + 1)
                    except Exception as frame_error:
                        logger.error(f"Error processing frame: {frame_error}")
            
            logger.info(f"Video processing completed. Processed {processed_count} frames out of {frame_count} total frames.")
            return True
        except Exception as e:
            logger.error(f"Error during video processing: {e}")
            return False
        finally:
            # Cleanup resources
            try:
                processor.close()
            except Exception as close_error:
                logger.error(f"Error closing processor: {close_error}")
            
            try:
                if writer:
                    writer.close()
            except Exception as close_error:
                logger.error(f"Error closing writer: {close_error}")
    
    def _process_video_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single video frame
        
        Args:
            frame: Input frame
        
        Returns:
            Dict[str, Any]: Processing results
        """
        # Run detection
        detections = self.detector.process(frame)
        
        # Run tracking if enabled
        tracks = []
        if self.enable_tracking and self.tracker is not None:
            tracks = self.tracker.process(detections, image=frame)
        
        # Run pose estimation if enabled
        poses = []
        if self.enable_pose_estimation and self.pose_estimator is not None:
            poses = self.pose_estimator.process(frame)
        
        return {
            "detections": detections,
            "tracks": tracks,
            "poses": poses
        }
    
    def _process_video_batch(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Process multiple video frames in batch
        
        Args:
            frames: List of input frames
        
        Returns:
            List[Dict[str, Any]]: List of processing results
        """
        from .batch import BatchPipeline
        batch_pipeline = BatchPipeline(self.config)
        batch_pipeline.initialize()
        return batch_pipeline.process_batch(frames)
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process a single image
        
        Args:
            image: Input image
        
        Returns:
            Dict[str, Any]: Processing results
        """
        return self._process_video_frame(image)
