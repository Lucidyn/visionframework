"""
Unified tracker interface

This module provides a unified interface for different object tracking algorithms,
including IoU Tracker and ByteTrack. Users can switch between algorithms by changing
the tracker_type configuration parameter.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .base import BaseModule
from ..data.detection import Detection
from ..data.track import Track
from ..utils.monitoring.logger import get_logger
from ..utils.data.trajectory_analyzer import TrajectoryAnalyzer

logger = get_logger(__name__)


class Tracker(BaseModule):
    """
    Unified tracker interface supporting multiple algorithms
    
    This class provides a unified interface for object tracking using different
    algorithms (IoU Tracker, ByteTrack, ReID Tracker). It automatically handles algorithm-specific
    initialization and provides a consistent API regardless of the underlying algorithm.
    
    Example:
        ```python
        # Using IoU Tracker
        tracker = Tracker({
            "tracker_type": "iou",
            "max_age": 30,
            "min_hits": 3,
            "iou_threshold": 0.3
        })
        tracker.initialize()
        tracks = tracker.update(detections)
        
        # Using ByteTrack
        tracker = Tracker({
            "tracker_type": "bytetrack",
            "track_thresh": 0.5,
            "track_buffer": 30
        })

        # Using ReID Tracker
        tracker = Tracker({
            "tracker_type": "reid",
            "reid_weight": 0.7,
            "reid_config": {
                "model_path": "path/to/reid_model.pth"
            }
        })
        ```
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize tracker
        
        Args:
            config: Configuration dictionary with keys:
                - tracker_type: Type of tracker, one of:
                    - 'iou': IoU Tracker (default) - Simple IoU-based tracking
                    - 'bytetrack': ByteTrack - Advanced tracking with high/low confidence detection association
                    - 'reid': ReID Tracker - Tracking with appearance features
                - max_age: Maximum frames a track can be missing before deletion (default: 30)
                  Must be a positive integer.
                - min_hits: Minimum number of consecutive detections to confirm a track (default: 3)
                  Must be a positive integer.
                - iou_threshold: IoU threshold for matching tracks with detections (default: 0.3)
                  Must be a number between 0.0 and 1.0.
                - track_thresh: ByteTrack confidence threshold for high-confidence detections (default: 0.5)
                  Must be a number between 0.0 and 1.0. Only used for ByteTrack.
                - track_buffer: ByteTrack buffer size for lost tracks (default: 30)
                  Must be a positive integer. Only used for ByteTrack.
                - match_thresh: ByteTrack matching threshold (default: 0.8)
                  Must be a number between 0.0 and 1.0. Only used for ByteTrack.
                - reid_weight: ReID weight for ReID Tracker (default: 0.7)
                - reid_config: ReID configuration dictionary
                - trajectory_analysis: Enable trajectory analysis (default: False)
                - fps: Frames per second for trajectory analysis (default: 30)
                - pixel_to_meter: Conversion factor from pixels to meters (default: None)
        
        Raises:
            ValueError: If configuration is invalid (will be logged as warning)
        """
        super().__init__(config)
        self.tracker_impl: Optional[BaseModule] = None
        self.tracker_type: str = self.config.get("tracker_type", "iou")
        
        # Initialize trajectory analyzer
        self.enable_trajectory_analysis: bool = self.config.get("trajectory_analysis", False)
        self.fps: float = self.config.get("fps", 30.0)
        self.pixel_to_meter: Optional[float] = self.config.get("pixel_to_meter", None)
        self.trajectory_analyzer: Optional[TrajectoryAnalyzer] = None
        
        if self.enable_trajectory_analysis:
            self.trajectory_analyzer = TrajectoryAnalyzer(fps=self.fps, pixel_to_meter=self.pixel_to_meter)
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate tracker configuration
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            Tuple[bool, Optional[str]]: 
                - (True, None) if valid
                - (False, error_message) if invalid
        """
        # Validate tracker_type
        if "tracker_type" in config:
            tracker_type = config["tracker_type"]
            if tracker_type not in ["iou", "bytetrack", "reid"]:
                return False, f"Invalid tracker_type: {tracker_type}. Supported: 'iou', 'bytetrack', 'reid'"
        
        # Validate max_age
        if "max_age" in config:
            max_age = config["max_age"]
            if not isinstance(max_age, int) or max_age <= 0:
                return False, f"max_age must be a positive integer, got {max_age}"
        
        # Validate min_hits
        if "min_hits" in config:
            min_hits = config["min_hits"]
            if not isinstance(min_hits, int) or min_hits <= 0:
                return False, f"min_hits must be a positive integer, got {min_hits}"
        
        # Validate iou_threshold
        if "iou_threshold" in config:
            iou_threshold = config["iou_threshold"]
            if not isinstance(iou_threshold, (int, float)):
                return False, f"iou_threshold must be a number, got {type(iou_threshold).__name__}"
            if not 0.0 <= iou_threshold <= 1.0:
                return False, f"iou_threshold must be between 0.0 and 1.0, got {iou_threshold}"
        
        # Validate ByteTrack-specific parameters
        if config.get("tracker_type") == "bytetrack" or "track_thresh" in config:
            if "track_thresh" in config:
                track_thresh = config["track_thresh"]
                if not isinstance(track_thresh, (int, float)):
                    return False, f"track_thresh must be a number, got {type(track_thresh).__name__}"
                if not 0.0 <= track_thresh <= 1.0:
                    return False, f"track_thresh must be between 0.0 and 1.0, got {track_thresh}"
            
            if "track_buffer" in config:
                track_buffer = config["track_buffer"]
                if not isinstance(track_buffer, int) or track_buffer <= 0:
                    return False, f"track_buffer must be a positive integer, got {track_buffer}"
            
            if "match_thresh" in config:
                match_thresh = config["match_thresh"]
                if not isinstance(match_thresh, (int, float)):
                    return False, f"match_thresh must be a number, got {type(match_thresh).__name__}"
                if not 0.0 <= match_thresh <= 1.0:
                    return False, f"match_thresh must be between 0.0 and 1.0, got {match_thresh}"

        # Validate ReID-specific parameters
        if config.get("tracker_type") == "reid":
            if "reid_weight" in config:
                reid_weight = config["reid_weight"]
                if not isinstance(reid_weight, (int, float)):
                    return False, f"reid_weight must be a number, got {type(reid_weight).__name__}"
                if not 0.0 <= reid_weight <= 1.0:
                    return False, f"reid_weight must be between 0.0 and 1.0, got {reid_weight}"
            
            if "reid_config" in config and not isinstance(config["reid_config"], dict):
                return False, f"reid_config must be a dictionary, got {type(config.get('reid_config')).__name__}"

        return True, None
    
    def initialize(self) -> bool:
        """
        Initialize the tracker algorithm

        This method initializes the underlying tracker implementation based on
        the configured tracker_type. It performs algorithm-specific setup steps
        and prepares the tracker for tracking objects.

        Returns:
            bool: True if initialization successful, False otherwise.
                  On failure, errors are logged with detailed information.

        Note:
            Initialization may involve:
            - Loading model files (for ReID tracker)
            - Setting up algorithm-specific parameters
            - Initializing internal state variables

        Example:
            ```python
            tracker = Tracker({"tracker_type": "iou"})
            if tracker.initialize():
                print("Tracker initialized successfully")
            else:
                print("Initialization failed, check logs for details")
            ```
        """
        try:
            # Lazy import tracker implementations to avoid importing heavy libraries at module load time
            if self.tracker_type == "iou":
                from .trackers.iou_tracker import IOUTracker
                self.tracker_impl = IOUTracker(self.config)
            elif self.tracker_type == "bytetrack":
                from .trackers.byte_tracker import ByteTracker
                self.tracker_impl = ByteTracker(self.config)
            elif self.tracker_type == "reid":
                from .trackers.reid_tracker import ReIDTracker
                self.tracker_impl = ReIDTracker(self.config)
            else:
                raise ValueError(f"Unsupported tracker_type: {self.tracker_type}. Supported: 'iou', 'bytetrack', 'reid'")
            
            # Initialize tracker_impl
            init_result = self.tracker_impl.initialize()
            
            # Set is_initialized only if tracker_impl initialization succeeded
            if init_result:
                self.is_initialized = True
            
            return init_result
        except ValueError as e:
            logger.error(f"Invalid tracker configuration: {e}", exc_info=True)
            return False
        except (ImportError, RuntimeError) as e:
            logger.error(f"Failed to initialize tracker ({self.tracker_type}): {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Unexpected error initializing tracker: {e}", exc_info=True)
            return False
    
    def process(self, detections: List[Detection], image: Optional[np.ndarray] = None) -> List[Track]:
        """
        Update tracker with new detections
        
        This method updates the tracker state with new detections from the current frame
        and returns the list of active tracks. If the tracker is not initialized,
        it will attempt to initialize automatically.
        
        Args:
            detections: List of Detection objects from the current frame.
                       Each Detection should contain at least:
                       - bbox: Tuple of (x1, y1, x2, y2) coordinates
                       - confidence: Confidence score
                       - class_id: Integer class ID
            image: Optional current frame image (needed for ReID trackers)
        
        Returns:
            List[Track]: List of Track objects representing currently tracked objects.
                        Each Track contains:
                        - track_id: Unique integer identifier for the track
                        - bbox: Current bounding box coordinates
                        - confidence: Current confidence score
                        - class_id: Object class ID
                        - class_name: Object class name
                        - age: Number of frames since track creation
                        - time_since_update: Frames since last update
                        - history: List of previous positions (if available)
            
            Returns empty list if:
                - Tracker is not initialized and initialization fails
                - tracker_impl is None
                - No tracks are active
                - An error occurs during tracking
        
        Raises:
            RuntimeError: If tracker is not initialized and automatic initialization fails
            ValueError: If detections format is invalid
        
        Example:
            ```python
            tracker = Tracker()
            tracker.initialize()
            
            # Process detections from each frame
            tracks = tracker.process(detections, frame)
            for track in tracks:
                print(f"Track ID {track.track_id}: {track.class_name}")
            ```
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("Tracker not initialized and auto-initialization failed")
                return []
        
        if self.tracker_impl is None:
            logger.warning("tracker_impl is None, returning empty list")
            return []
        
        try:
            tracks = self.tracker_impl.update(detections, image=image)
            
            # Convert STrack to Track if needed (ByteTrack and ReID Tracker return STrack objects)
            if self.tracker_type in ["bytetrack", "reid"]:
                from ..data.track import STrack
                if tracks and len(tracks) > 0 and isinstance(tracks[0], STrack):
                    return [st.to_track() for st in tracks]
            
            return tracks
        except Exception as e:
            logger.error(f"Error during tracking: {e}", exc_info=True)
            return []
    
    def update(self, detections: List[Detection], image: Optional[np.ndarray] = None) -> List[Track]:
        """
        Alias for process method
        
        This method is provided for convenience and clarity. It is functionally
        equivalent to process().
        
        Args:
            detections: List of Detection objects from the current frame
            image: Optional current frame image
        
        Returns:
            List[Track]: List of tracked objects
        """
        return self.process(detections, image=image)    
    def process_batch(self, 
                     detections_list: List[List[Detection]], 
                     images: Optional[List[np.ndarray]] = None) -> List[List[Track]]:
        """
        Process multiple frames of detections through tracker sequentially.
        
        This method updates the tracker state with detections from multiple frames
        and returns the active tracks for each frame. Unlike detectors, trackers
        maintain state across frames, so batch processing means processing frames
        sequentially while maintaining temporal coherence.
        
        Args:
            detections_list: List of detection lists, one per frame
            images: Optional list of frame images (needed for some trackers like ReID tracker)
        
        Returns:
            List[List[Track]]: List of track lists, one per frame
        
        Example:
            ```python
            tracker = Tracker()
            tracker.initialize()
            
            # Process detections from multiple frames sequentially
            detections_frames = [dets_frame1, dets_frame2, dets_frame3]
            images_frames = [frame1, frame2, frame3]
            
            tracks_frames = tracker.process_batch(detections_frames, images_frames)
            
            for frame_idx, tracks in enumerate(tracks_frames):
                print(f"Frame {frame_idx}: {len(tracks)} active tracks")
            ```
        """
        if not self.is_initialized:
            if not self.initialize():
                logger.error("Tracker not initialized and auto-initialization failed")
                return [[] for _ in detections_list]
        
        if self.tracker_impl is None:
            logger.warning("tracker_impl is None, returning empty lists")
            return [[] for _ in detections_list]
        
        batch_tracks: List[List[Track]] = []
        
        try:
            # Process each frame sequentially to maintain temporal coherence
            for frame_idx, detections in enumerate(detections_list):
                image = images[frame_idx] if images and frame_idx < len(images) else None
                tracks = self.process(detections, image)
                batch_tracks.append(tracks)
            
            return batch_tracks
        except Exception as e:
            logger.error(f"Error during batch tracking: {e}", exc_info=True)
            return [[] for _ in detections_list]
    
    def update_batch(self, 
                    detections_list: List[List[Detection]], 
                    images: Optional[List[np.ndarray]] = None) -> List[List[Track]]:
        """
        Alias for process_batch method
        
        This method is provided for convenience and clarity. It is functionally
        equivalent to process_batch().
        
        Args:
            detections_list: List of detection lists, one per frame
            images: Optional list of frame images
        
        Returns:
            List[List[Track]]: List of track lists, one per frame
        """
        return self.process_batch(detections_list, images)    
    def reset(self) -> None:
        """
        Reset tracker state
        
        This method resets the tracker to its initial state, clearing all
        active tracks and resetting internal counters. The tracker remains
        initialized but will start fresh with no tracks.
        
        Subclasses that override this method should call super().reset() first.
        """
        super().reset()
        if self.tracker_impl and hasattr(self.tracker_impl, 'reset'):
            self.tracker_impl.reset()
    
    def get_tracks(self) -> List[Track]:
        """
        Get all active tracks
        
        Returns a list of all currently active (confirmed) tracks without
        processing new detections. This is useful for querying track state
        between frames.
        
        Returns:
            List[Track]: List of all active Track objects.
                        Returns empty list if:
                        - Tracker is not initialized
                        - tracker_impl does not support get_tracks()
                        - No tracks are currently active
        
        Note:
            This method only returns confirmed tracks. Lost or unconfirmed
            tracks are not included in the result.
        
        Example:
            ```python
            # Get current active tracks without updating
            active_tracks = tracker.get_tracks()
            print(f"Currently tracking {len(active_tracks)} objects")
            ```
        """
        if not self.is_initialized or self.tracker_impl is None:
            return []
        
        if hasattr(self.tracker_impl, 'get_tracks'):
            try:
                tracks = self.tracker_impl.get_tracks()
                # Convert STrack to Track if needed
                if tracks and len(tracks) > 0 and hasattr(tracks[0], 'to_track'):
                    return [st.to_track() for st in tracks]
                return tracks if tracks else []
            except Exception as e:
                logger.error(f"Error getting tracks: {e}", exc_info=True)
                return []
        
        return []
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """
        Get track by ID
        
        Retrieves a specific track by its unique identifier.
        
        Args:
            track_id: Integer track ID to search for
        
        Returns:
            Optional[Track]: Track object with the specified ID if found, None otherwise.
                           Returns None if:
                           - Tracker is not initialized
                           - tracker_impl does not support get_track_by_id()
                           - Track with the specified ID does not exist
        
        Example:
            ```python
            # Find a specific track
            track = tracker.get_track_by_id(track_id=5)
            if track:
                print(f"Track 5: {track.class_name} at {track.bbox}")
            ```
        """
        if not self.is_initialized or self.tracker_impl is None:
            return None
        
        if hasattr(self.tracker_impl, 'get_track_by_id'):
            try:
                track = self.tracker_impl.get_track_by_id(track_id)
                # Convert STrack to Track if needed
                if track and hasattr(track, 'to_track'):
                    return track.to_track()
                return track
            except Exception as e:
                logger.error(f"Error getting track by ID: {e}", exc_info=True)
                return None
        
        return None
    
    def analyze_track(self, track: Track) -> Optional[Dict[str, Any]]:
        """
        Analyze a single track's trajectory
        
        Args:
            track: Track object to analyze
        
        Returns:
            Optional[Dict[str, Any]]: Dictionary containing trajectory analysis results
                                      Returns None if trajectory analysis is disabled.
        
        Example:
            ```python
            track = tracker.get_track_by_id(5)
            if track:
                analysis = tracker.analyze_track(track)
                if analysis:
                    print(f"Track {track.track_id} speed: {analysis['speed']['magnitude']:.2f} pixels/frame")
            ```
        """
        if not self.enable_trajectory_analysis or self.trajectory_analyzer is None:
            return None
        
        try:
            return self.trajectory_analyzer.analyze_track(track)
        except Exception as e:
            logger.error(f"Error analyzing track: {e}", exc_info=True)
            return None
    
    def analyze_tracks(self, tracks: Optional[List[Track]] = None) -> List[Dict[str, Any]]:
        """
        Analyze trajectories for multiple tracks
        
        Args:
            tracks: List of tracks to analyze. If None, analyze all active tracks.
        
        Returns:
            List[Dict[str, Any]]: List of trajectory analysis results
        
        Example:
            ```python
            # Analyze all active tracks
            analysis_results = tracker.analyze_tracks()
            for result in analysis_results:
                print(f"Track {result['track_id']}: Speed={result['speed']['magnitude']:.2f}")
            
            # Analyze specific tracks
            specific_tracks = [track1, track2]
            analysis_results = tracker.analyze_tracks(specific_tracks)
            ```
        """
        if not self.enable_trajectory_analysis or self.trajectory_analyzer is None:
            return []
        
        try:
            # If no tracks provided, get all active tracks
            if tracks is None:
                tracks = self.get_tracks()
            
            return self.trajectory_analyzer.analyze_tracks(tracks)
        except Exception as e:
            logger.error(f"Error analyzing tracks: {e}", exc_info=True)
            return []
    
    def smooth_track_trajectory(self, track: Track, window_size: int = 5) -> List[Tuple[float, float, float, float]]:
        """
        Smooth a track's trajectory using moving average
        
        Args:
            track: Track object to smooth
            window_size: Size of moving average window
        
        Returns:
            List[Tuple[float, float, float, float]]: List of smoothed bounding boxes
        
        Example:
            ```python
            track = tracker.get_track_by_id(5)
            if track:
                smoothed = tracker.smooth_track_trajectory(track, window_size=3)
                print(f"Original trajectory length: {len(track.history)}")
                print(f"Smoothed trajectory length: {len(smoothed)}")
            ```
        """
        if not self.enable_trajectory_analysis or self.trajectory_analyzer is None:
            return track.history
        
        try:
            return self.trajectory_analyzer.smooth_trajectory(track, window_size)
        except Exception as e:
            logger.error(f"Error smoothing track trajectory: {e}", exc_info=True)
            return track.history
    
    def predict_next_position(self, track: Track, frames_ahead: int = 1) -> Tuple[float, float, float, float]:
        """
        Predict a track's next position using linear extrapolation
        
        Args:
            track: Track object to predict
            frames_ahead: Number of frames to predict ahead
        
        Returns:
            Tuple[float, float, float, float]: Predicted bounding box
        
        Example:
            ```python
            track = tracker.get_track_by_id(5)
            if track:
                predicted_bbox = tracker.predict_next_position(track, frames_ahead=2)
                print(f"Current bbox: {track.bbox}")
                print(f"Predicted bbox in 2 frames: {predicted_bbox}")
            ```
        """
        if not self.enable_trajectory_analysis or self.trajectory_analyzer is None:
            return track.bbox
        
        try:
            return self.trajectory_analyzer.predict_next_position(track, frames_ahead)
        except Exception as e:
            logger.error(f"Error predicting next position: {e}", exc_info=True)
            return track.bbox
    
    def cleanup(self) -> None:
        """Cleanup resources held by tracker and underlying implementation."""
        try:
            if self.tracker_impl and hasattr(self.tracker_impl, 'cleanup'):
                try:
                    self.tracker_impl.cleanup()
                except Exception as e:
                    logger.warning(f"Error during tracker_impl.cleanup(): {e}")
        finally:
            self.tracker_impl = None
            self.is_initialized = False
