"""
Video processing utilities
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Callable, Any, Dict
from pathlib import Path


class VideoProcessor:
    """Video processing utilities"""
    
    def __init__(self, video_path: str):
        """
        Initialize video processor
        
        Args:
            video_path: Path to video file or camera index (0 for webcam)
        """
        self.video_path = video_path
        self.cap = None
        self.is_camera = isinstance(video_path, int) or video_path == "0"
        self.fps = 0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.current_frame_num = 0
    
    def open(self) -> bool:
        """Open video file or camera"""
        try:
            if self.is_camera:
                self.cap = cv2.VideoCapture(int(self.video_path) if isinstance(self.video_path, str) else self.video_path)
            else:
                self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                return False
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if not self.is_camera:
                self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                self.total_frames = -1  # Unknown for camera
            
            return True
        except Exception as e:
            print(f"Failed to open video: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame
        
        Returns:
            Tuple of (success, frame)
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_num += 1
        return ret, frame
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get specific frame by frame number
        
        Args:
            frame_number: Frame number to get
            
        Returns:
            Frame or None if failed
        """
        if self.cap is None:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def get_info(self) -> Dict[str, Any]:
        """Get video information"""
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "current_frame": self.current_frame_num,
            "is_camera": self.is_camera
        }
    
    def close(self):
        """Close video"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class VideoWriter:
    """Video writer utility"""
    
    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        frame_size: Optional[Tuple[int, int]] = None,
        fourcc: str = "mp4v"
    ):
        """
        Initialize video writer
        
        Args:
            output_path: Output video path
            fps: Frames per second
            frame_size: (width, height) of frames
            fourcc: FourCC codec (e.g., 'mp4v', 'XVID', 'H264')
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = None
    
    def open(self, frame_size: Optional[Tuple[int, int]] = None) -> bool:
        """
        Open video writer
        
        Args:
            frame_size: Frame size if not set in __init__
            
        Returns:
            bool: True if successful
        """
        try:
            size = frame_size or self.frame_size
            if size is None:
                raise ValueError("Frame size must be specified")
            
            self.writer = cv2.VideoWriter(
                self.output_path,
                self.fourcc,
                self.fps,
                size
            )
            
            return self.writer.isOpened()
        except Exception as e:
            print(f"Failed to open video writer: {e}")
            return False
    
    def write(self, frame: np.ndarray) -> bool:
        """
        Write frame to video
        
        Args:
            frame: Frame to write
            
        Returns:
            bool: True if successful
        """
        if self.writer is None:
            return False
        
        # Auto-detect frame size on first write
        if self.frame_size is None:
            h, w = frame.shape[:2]
            self.frame_size = (w, h)
            if not self.open():
                return False
        
        self.writer.write(frame)
        return True
    
    def close(self):
        """Close video writer"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def process_video(
    input_path: str,
    output_path: Optional[str] = None,
    frame_callback: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    skip_frames: int = 0
) -> bool:
    """
    Process video with callback function
    
    Args:
        input_path: Input video path or camera index
        output_path: Optional output video path
        frame_callback: Function(frame, frame_number) -> processed_frame
        start_frame: Start frame number
        end_frame: End frame number (None for all)
        skip_frames: Number of frames to skip between processing
        
    Returns:
        bool: True if successful
    """
    processor = VideoProcessor(input_path)
    
    if not processor.open():
        return False
    
    writer = None
    if output_path:
        info = processor.get_info()
        writer = VideoWriter(output_path, fps=info["fps"], frame_size=(info["width"], info["height"]))
        if not writer.open():
            writer = None
    
    try:
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = processor.read_frame()
            if not ret:
                break
            
            # Skip frames
            if frame_count < start_frame or (end_frame and frame_count > end_frame):
                frame_count += 1
                continue
            
            if skip_frames > 0 and (frame_count - start_frame) % (skip_frames + 1) != 0:
                frame_count += 1
                continue
            
            # Process frame
            if frame_callback:
                processed_frame = frame_callback(frame, frame_count)
            else:
                processed_frame = frame
            
            # Write frame
            if writer:
                writer.write(processed_frame)
            
            processed_count += 1
            frame_count += 1
        
        return True
    except Exception as e:
        print(f"Error processing video: {e}")
        return False
    finally:
        processor.close()
        if writer:
            writer.close()

