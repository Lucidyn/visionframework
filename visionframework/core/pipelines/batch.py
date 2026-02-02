"""
Batch pipeline class for Vision Framework
"""

import time
import numpy as np
from typing import Dict, Any, List, Optional
from .base import BasePipeline
# Detector and Tracker imports are now handled via components
from ...data.detection import Detection
from ...data.track import Track
from ...data.pose import Pose
from ...utils.monitoring.logger import get_logger
from ...utils.concurrent import parallel_map
from ...utils.memory import optimize_memory_usage

logger = get_logger(__name__)


class BatchPipeline(BasePipeline):
    """
    Batch pipeline for processing multiple images efficiently
    
    This class optimizes the processing of multiple images by batching them
    through the detector when possible, providing significant performance improvements.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize batch pipeline
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Set default config values
        self.config.setdefault("detector_config", {
            "model_path": "yolov8n.pt",
            "conf_threshold": 0.25,
            "batch_inference": True
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
        Initialize batch pipeline components
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize base pipeline
            if not super().initialize():
                return False
            
            # Initialize detector
            logger.info("Initializing detector for batch processing...")
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
            
            logger.info("Batch pipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize batch pipeline: {e}")
            return False
    
    def process_batch(self, images: List[np.ndarray], max_batch_size: Optional[int] = None, use_parallel: bool = False, max_workers: Optional[int] = None, enable_memory_optimization: bool = True) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch
        
        Args:
            images: List of input images
            max_batch_size: Maximum batch size for processing
            use_parallel: Whether to use parallel processing
            max_workers: Maximum number of workers for parallel processing
            enable_memory_optimization: Whether to enable memory optimization
        
        Returns:
            List[Dict[str, Any]]: List of processing results
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("Pipeline not initialized and auto-initialization failed")
                return [{"detections": [], "tracks": [], "poses": [], "frame_idx": i, "processing_time": 0.0} for i in range(len(images))]
        
        if self.detector is None:
            logger.error("Detector is None, cannot process batch")
            return [{"detections": [], "tracks": [], "poses": [], "frame_idx": i, "processing_time": 0.0} for i in range(len(images))]
        
        if not images:
            logger.warning("Empty image list provided to process_batch")
            return []
        
        try:
            # Optimize memory usage if enabled
            if enable_memory_optimization:
                optimize_memory_usage()
            
            # Dynamic batch size adjustment based on image size and count
            if max_batch_size is None:
                if len(images) > 32:
                    max_batch_size = 16
                elif len(images) > 16:
                    max_batch_size = 8
                else:
                    max_batch_size = len(images)
                logger.debug(f"Auto-determined batch size: {max_batch_size}")
            
            # Split into smaller batches if max_batch_size is specified
            if max_batch_size and len(images) > max_batch_size:
                logger.debug(f"Splitting {len(images)} images into batches of {max_batch_size}")
                chunk_results = []
                total_processing_time = 0.0
                
                for i in range(0, len(images), max_batch_size):
                    chunk = images[i:i + max_batch_size]
                    start_time = time.time()
                    chunk_result = self._process_batch_chunk(chunk, i, use_parallel, max_workers, enable_memory_optimization)
                    chunk_time = time.time() - start_time
                    total_processing_time += chunk_time
                    chunk_results.extend(chunk_result)
                    
                    # Memory optimization between chunks
                    if enable_memory_optimization:
                        optimize_memory_usage()
                    
                    logger.debug(f"Processed chunk {i//max_batch_size + 1}/{(len(images) + max_batch_size - 1)//max_batch_size} in {chunk_time:.4f}s")
                
                logger.info(f"Batch processing completed: {len(images)} images in {total_processing_time:.4f}s, average {total_processing_time/len(images):.4f}s per image")
                return chunk_results
            else:
                # Process all images in one batch
                start_time = time.time()
                results = self._process_batch_chunk(images, 0, use_parallel, max_workers, enable_memory_optimization)
                total_time = time.time() - start_time
                
                logger.info(f"Batch processing completed: {len(images)} images in {total_time:.4f}s, average {total_time/len(images):.4f}s per image")
                return results
        except Exception as e:
            logger.error(f"Error during batch pipeline processing: {e}")
            error_results = []
            for i in range(len(images)):
                result = {
                    "detections": [], 
                    "tracks": [], 
                    "poses": [], 
                    "frame_idx": i, 
                    "processing_time": 0.0,
                    "error": str(e)
                }
                error_results.append(result)
            return error_results
        finally:
            # Ensure memory cleanup
            if enable_memory_optimization:
                optimize_memory_usage()
    
    def _process_batch_chunk(self, images: List[np.ndarray], start_frame_idx: int = 0, use_parallel: bool = False, max_workers: Optional[int] = None, enable_memory_optimization: bool = True) -> List[Dict[str, Any]]:
        """
        Process a chunk of images in batch
        
        Args:
            images: List of input images
            start_frame_idx: Starting frame index for this chunk
            use_parallel: Whether to use parallel processing
            max_workers: Maximum number of workers for parallel processing
            enable_memory_optimization: Whether to enable memory optimization
            
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
            start_time = time.time()
            all_detections = self.detector.process(images)
            batch_processing_time = time.time() - start_time
            per_frame_time = batch_processing_time / len(images)
            
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
                    start_time = time.time()
                    dets = self.detector.process(image)
                    processing_time = time.time() - start_time
                    return dets, processing_time
                
                results_with_time = parallel_map(
                    images, 
                    process_single_image, 
                    max_workers=max_workers,
                    use_processes=False  # Use threads for I/O-bound operations
                )
                detections_per_image = [result[0] for result in results_with_time]
                processing_times = [result[1] for result in results_with_time]
            else:
                # Process sequentially
                detections_per_image = []
                processing_times = []
                for image in images:
                    start_time = time.time()
                    dets = self.detector.process(image)
                    processing_time = time.time() - start_time
                    detections_per_image.append(dets)
                    processing_times.append(processing_time)
        
        # Process tracking and pose estimation for each frame
        if use_parallel and len(images) > 1 and not self.enable_tracking:
            # Use parallel processing for post-processing (if no tracking)
            logger.debug(f"Using parallel post-processing")
            
            def process_post(image_and_detections):
                image, detections, frame_idx, processing_time = image_and_detections
                frame_result: Dict[str, Any] = {
                    "detections": detections,
                    "frame_idx": start_frame_idx + frame_idx,
                    "processing_time": processing_time
                }
                
                # Run pose estimation if enabled
                if self.enable_pose_estimation and self.pose_estimator is not None:
                    poses = self.pose_estimator.process(image)
                    frame_result["poses"] = poses
                else:
                    frame_result["poses"] = []
                
                return frame_result
            
            # Prepare data for parallel processing
            if use_batch:
                # For batch processing, use the same time for all frames
                data_for_parallel = [(img, dets, idx, per_frame_time) for idx, (img, dets) in enumerate(zip(images, detections_per_image))]
            else:
                # For individual processing, use per-frame times
                data_for_parallel = [(img, dets, idx, processing_times[idx]) for idx, (img, dets) in enumerate(zip(images, detections_per_image))]
            
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
                # Get processing time for this frame
                if use_batch:
                    processing_time = per_frame_time
                else:
                    processing_time = processing_times[frame_idx] if frame_idx < len(processing_times) else 0.0
                
                frame_result: Dict[str, Any] = {
                    "detections": detections,
                    "frame_idx": start_frame_idx + frame_idx,
                    "processing_time": processing_time
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
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process a single image
        
        Args:
            image: Input image
        
        Returns:
            Dict[str, Any]: Processing results
        """
        results = self.process_batch([image])
        return results[0] if results else {"detections": [], "tracks": [], "poses": []}
