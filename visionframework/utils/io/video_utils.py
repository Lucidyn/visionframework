"""
Video processing utilities

This module provides comprehensive video processing tools for reading, writing, and processing video files
and camera streams. It includes classes for video reading, writing, and a high-level function for
processing videos with custom callbacks.

Main components:
- VideoProcessor: For reading video files and camera streams
- VideoWriter: For writing processed frames to video files
- process_video: High-level function for processing videos with callbacks

Example usage:
    ```python
    from visionframework.utils import VideoProcessor, VideoWriter, process_video
    
    # Read and process video frames
    processor = VideoProcessor("input.mp4")
    if processor.open():
        while True:
            ret, frame = processor.read_frame()
            if not ret:
                break
            # Process frame here
            
        processor.close()
    
    # Write processed frames to video
    writer = VideoWriter("output.mp4", fps=30.0, frame_size=(640, 480))
    if writer.open():
        writer.write(frame)  # Write processed frame
        writer.close()
    
    # Use high-level process_video function
    def frame_callback(frame, frame_number):
        # Process frame here
        return frame
    
    success = process_video(
        input_path="input.mp4",
        output_path="output.mp4",
        frame_callback=frame_callback
    )
    ```
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Callable, Any, Dict
from pathlib import Path

# Try to import pyav, but keep it optional
try:
    import av
except ImportError:
    av = None


class VideoProcessor:
    """Video processing class for reading video files and camera streams
    
    This class provides an easy-to-use interface for reading video frames from files or cameras,
    with support for frame seeking, video information retrieval, and context manager usage.
    
    Example:
        ```python
        # Read video file using context manager
        from visionframework.utils import VideoProcessor
        
        with VideoProcessor("input.mp4") as processor:
            info = processor.get_info()
            print(f"Video info: {info}")
            
            while True:
                ret, frame = processor.read_frame()
                if not ret:
                    break
                # Process frame here
                print(f"Processing frame {processor.current_frame_num}")
        ```
    """
    
    def __init__(self, video_path: str or int):
        """
        Initialize video processor
        
        Args:
            video_path: Path to video file, video stream URL, or camera index (integer, 0 for default webcam)
            Supported stream formats:
            - RTSP: rtsp://example.com/stream
            - HTTP: http://example.com/stream
            - Camera: 0, 1, etc.
            - Video files: .mp4, .avi, .mov, .mkv, etc.
        
        Examples:
            ```python
            # Initialize with video file
            processor = VideoProcessor("input.mp4")
            
            # Initialize with camera
            processor = VideoProcessor(0)  # Default webcam
            
            # Initialize with RTSP stream
            processor = VideoProcessor("rtsp://username:password@example.com/stream")
            
            # Initialize with HTTP stream
            processor = VideoProcessor("http://example.com/stream.mjpg")
            ```
        """
        self.video_path = video_path
        self.cap = None
        # Check if it's a camera (integer or string "0") or a video stream/file
        self.is_camera = isinstance(video_path, int) or (isinstance(video_path, str) and video_path == "0")
        # Check if it's a video stream URL
        self.is_stream = isinstance(video_path, str) and (video_path.startswith("rtsp://") or \
                                                         video_path.startswith("http://") or \
                                                         video_path.startswith("https://"))
        self.fps = 0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.current_frame_num = 0
    
    def open(self) -> bool:
        """
        Open video file or camera
        
        Returns:
            bool: True if successful, False otherwise
        
        Example:
            ```python
            processor = VideoProcessor("input.mp4")
            if processor.open():
                print("Video opened successfully")
            else:
                print("Failed to open video")
            ```
        """
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
        Read next frame from the video stream
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
                - success: True if frame read successfully, False if end of stream
                - frame: Numpy array containing the frame in BGR format, or None if failed
        
        Example:
            ```python
            ret, frame = processor.read_frame()
            if ret:
                print(f"Frame shape: {frame.shape}")
            else:
                print("End of video")
            ```
        """
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame_num += 1
        return ret, frame
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get specific frame by frame number (random access)
        
        Args:
            frame_number: Frame number to retrieve (0-indexed)
            
        Returns:
            Optional[np.ndarray]: Frame as numpy array in BGR format, or None if failed
        
        Example:
            ```python
            # Get the 100th frame
            frame = processor.get_frame(99)
            if frame is not None:
                print(f"Retrieved frame shape: {frame.shape}")
            ```
        """
        if self.cap is None:
            return None
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get video information and properties
        
        Returns:
            Dict[str, Any]: Dictionary containing video properties:
                - fps: Frames per second
                - width: Video width in pixels
                - height: Video height in pixels
                - total_frames: Total number of frames (-1 for camera/stream)
                - current_frame: Current frame number
                - is_camera: True if source is camera, False if video file or stream
                - is_stream: True if source is a video stream (RTSP/HTTP), False otherwise
        
        Example:
            ```python
            info = processor.get_info()
            print(f"Video resolution: {info['width']}x{info['height']}")
            print(f"FPS: {info['fps']}")
            print(f"Source type: {'Camera' if info['is_camera'] else 'Stream' if info['is_stream'] else 'File'}")
            ```
        """
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "current_frame": self.current_frame_num,
            "is_camera": self.is_camera,
            "is_stream": self.is_stream
        }
    
    def close(self):
        """
        Close video file or camera stream
        
        Example:
            ```python
            processor = VideoProcessor("input.mp4")
            processor.open()
            # Process frames...
            processor.close()  # Release resources
            ```
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def __enter__(self):
        """Context manager entry point - opens the video"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point - closes the video"""
        self.close()
        return False


class VideoWriter:
    """Video writer class for saving processed frames to video files
    
    This class provides an easy-to-use interface for writing video frames to files,
    supporting auto-detection of frame size and context manager usage.
    
    Example:
        ```python
        from visionframework.utils import VideoWriter
        
        # Write frames using context manager
        with VideoWriter("output.mp4", fps=30.0, frame_size=(640, 480)) as writer:
            for frame in frames:
                writer.write(frame)
        ```
    """
    
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
            output_path: Path to save the output video file
            fps: Frames per second (default: 30.0)
            frame_size: Tuple of (width, height) for the output video (default: None, auto-detected)
            fourcc: FourCC codec string (default: 'mp4v', other options: 'XVID', 'H264')
        
        Examples:
            ```python
            # Initialize with frame size
            writer = VideoWriter("output.mp4", fps=30.0, frame_size=(640, 480))
            
            # Initialize without frame size (auto-detected on first write)
            writer = VideoWriter("output.mp4", fps=30.0)
            ```
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = None
    
    def open(self, frame_size: Optional[Tuple[int, int]] = None) -> bool:
        """
        Open video writer for writing
        
        Args:
            frame_size: Optional frame size override (width, height)
            
        Returns:
            bool: True if writer opened successfully, False otherwise
        
        Example:
            ```python
            writer = VideoWriter("output.mp4", fps=30.0)
            if writer.open((640, 480)):
                print("Video writer opened successfully")
            else:
                print("Failed to open video writer")
            ```
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
        Write a single frame to the video file
        
        Args:
            frame: Numpy array containing the frame in BGR format
            
        Returns:
            bool: True if frame written successfully, False otherwise
        
        Example:
            ```python
            # Write a frame
            if writer.write(frame):
                print("Frame written successfully")
            else:
                print("Failed to write frame")
            ```
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
        """
        Close video writer and release resources
        
        Example:
            ```python
            writer = VideoWriter("output.mp4")
            writer.open((640, 480))
            # Write frames...
            writer.close()  # Release resources
            ```
        """
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def __enter__(self):
        """Context manager entry point"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point - closes the writer"""
        self.close()
        return False


class PyAVVideoProcessor:
    """Video processing class for reading video files and streams using pyav
    
    This class provides an easy-to-use interface for reading video frames from files
    or streams using pyav, which is based on FFmpeg and typically offers better performance
    than OpenCV's VideoCapture.
    
    Example:
        ```python
        # Read video file using context manager
        from visionframework.utils import PyAVVideoProcessor
        
        with PyAVVideoProcessor("input.mp4") as processor:
            info = processor.get_info()
            print(f"Video info: {info}")
            
            while True:
                ret, frame = processor.read_frame()
                if not ret:
                    break
                # Process frame here
                print(f"Processing frame {processor.current_frame_num}")
        
        # Read RTSP stream using context manager
        with PyAVVideoProcessor("rtsp://example.com/stream") as processor:
            info = processor.get_info()
            print(f"Stream info: {info}")
            
            while True:
                ret, frame = processor.read_frame()
                if not ret:
                    break
                # Process frame here
                print(f"Processing frame {processor.current_frame_num}")
        ```
    """
    
    def __init__(self, video_path: str):
        """
        Initialize pyav video processor
        
        Args:
            video_path: Path to video file or RTSP/HTTP stream URL
        """
        if av is None:
            raise ImportError("PyAV is required for PyAVVideoProcessor. Install with: pip install av")
        
        self.video_path = video_path
        self.container = None
        self.stream = None
        self.decoder = None
        self.fps = 0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.current_frame_num = 0
        self.frame_generator = None
        # Check if it's a video stream URL
        self.is_stream = isinstance(video_path, str) and (video_path.startswith("rtsp://") or \
                                                         video_path.startswith("http://") or \
                                                         video_path.startswith("https://"))
    
    def open(self) -> bool:
        """
        Open video file or stream using pyav
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Open the video container or stream
            self.container = av.open(self.video_path)
            
            # Get the first video stream
            self.stream = next(s for s in self.container.streams if s.type == 'video')
            
            # Get video properties
            self.fps = self.stream.average_rate or 30.0  # Default to 30fps for streams
            self.width = self.stream.width
            self.height = self.stream.height
            
            # For streams, total_frames is not available
            if self.is_stream:
                self.total_frames = -1  # -1 indicates unknown total frames
            else:
                self.total_frames = self.stream.frames or 0
            
            # Create frame generator
            self.frame_generator = self.container.decode(video=0)
            
            return True
        except Exception as e:
            print(f"Failed to open video with pyav: {e}")
            return False
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read next frame from the video
        
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
                - success: True if frame read successfully, False if end of stream
                - frame: Numpy array containing the frame in BGR format, or None if failed
        """
        if self.frame_generator is None:
            return False, None
        
        try:
            frame = next(self.frame_generator)
            # Convert to numpy array (RGB format)
            img = frame.to_rgb().to_ndarray()
            # Convert RGB to BGR for compatibility with OpenCV
            img = img[..., ::-1]
            # Ensure the array is contiguous for OpenCV compatibility
            if not img.flags.contiguous:
                img = np.ascontiguousarray(img)
            self.current_frame_num += 1
            return True, img
        except StopIteration:
            # End of video
            return False, None
        except Exception as e:
            print(f"Error reading frame: {e}")
            return False, None
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Get specific frame by frame number (random access)
        
        Args:
            frame_number: Frame number to retrieve (0-indexed)
            
        Returns:
            Optional[np.ndarray]: Frame as numpy array in BGR format, or None if failed
        """
        if self.container is None:
            return None
        
        try:
            # Seek to the frame
            self.container.seek(frame_number * int(1e6 / self.fps), whence='time')
            
            # Reset frame generator
            self.frame_generator = self.container.decode(video=0)
            
            # Read the frame
            ret, frame = self.read_frame()
            return frame if ret else None
        except Exception as e:
            print(f"Error seeking frame: {e}")
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get video information and properties
        
        Returns:
            Dict[str, Any]: Dictionary containing video properties
        """
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "current_frame": self.current_frame_num,
            "is_camera": False,
            "is_stream": self.is_stream,
            "backend": "pyav"
        }
    
    def close(self):
        """
        Close video file or stream
        """
        if self.container is not None:
            self.container.close()
            self.container = None
            self.stream = None
            self.decoder = None
            self.frame_generator = None
    
    def __enter__(self):
        """Context manager entry point - opens the video"""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point - closes the video"""
        self.close()
        return False


class PyAVVideoWriter:
    """Video writer class for saving processed frames to video files using pyav
    
    This class provides an easy-to-use interface for writing video frames to files
    using pyav, which is based on FFmpeg and typically offers better performance
    and more codec options than OpenCV's VideoWriter.
    
    Example:
        ```python
        from visionframework.utils import PyAVVideoWriter
        
        # Write frames using context manager
        with PyAVVideoWriter("output.mp4", fps=30.0, frame_size=(640, 480)) as writer:
            for frame in frames:
                writer.write(frame)
        ```
    """
    
    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        frame_size: Optional[Tuple[int, int]] = None,
        codec: str = "h264"
    ):
        """
        Initialize pyav video writer
        
        Args:
            output_path: Path to save the output video file
            fps: Frames per second (default: 30.0)
            frame_size: Tuple of (width, height) for the output video (default: None, auto-detected)
            codec: Video codec to use (default: "h264")
        """
        if av is None:
            raise ImportError("PyAV is required for PyAVVideoWriter. Install with: pip install av")
        
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.container = None
        self.stream = None
        self.frame_count = 0
    
    def open(self, frame_size: Optional[Tuple[int, int]] = None) -> bool:
        """
        Open video writer for writing
        
        Args:
            frame_size: Optional frame size override (width, height)
            
        Returns:
            bool: True if writer opened successfully, False otherwise
        """
        try:
            size = frame_size or self.frame_size
            if size is None:
                raise ValueError("Frame size must be specified")
            
            # Create output container
            self.container = av.open(self.output_path, mode='w')
            
            # Add video stream
            self.stream = self.container.add_stream(self.codec, rate=self.fps)
            self.stream.width = size[0]
            self.stream.height = size[1]
            self.stream.pix_fmt = "yuv420p"
            
            return True
        except Exception as e:
            print(f"Failed to open video writer with pyav: {e}")
            return False
    
    def write(self, frame: np.ndarray) -> bool:
        """
        Write a single frame to the video file
        
        Args:
            frame: Numpy array containing the frame in BGR format
            
        Returns:
            bool: True if frame written successfully, False otherwise
        """
        if self.container is None:
            # Auto-detect frame size on first write
            h, w = frame.shape[:2]
            if not self.open((w, h)):
                return False
        
        try:
            # Convert BGR to RGB
            frame_rgb = frame[..., ::-1]
            
            # Create pyav frame
            av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
            
            # Encode and write frame
            for packet in self.stream.encode(av_frame):
                self.container.mux(packet)
            
            self.frame_count += 1
            return True
        except Exception as e:
            print(f"Error writing frame: {e}")
            return False
    
    def close(self):
        """
        Close video writer and flush remaining packets
        """
        if self.container is not None:
            # Flush remaining packets
            if self.stream:
                for packet in self.stream.encode():
                    self.container.mux(packet)
            self.container.close()
            self.container = None
            self.stream = None
    
    def __enter__(self):
        """Context manager entry point"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point - closes the writer"""
        self.close()
        return False


def process_video(
    input_path: str or int,
    output_path: Optional[str] = None,
    frame_callback: Optional[Callable[[np.ndarray, int], np.ndarray]] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    skip_frames: int = 0,
    use_pyav: bool = False
) -> bool:
    """
    High-level function for processing video files or camera streams with a callback
    
    This function handles the entire video processing pipeline, including opening the video,
    processing frames with the provided callback, and saving the result if an output path is provided.
    
    Args:
        input_path: Input video path (string) or camera index (integer)
        output_path: Optional output video path. If None, video is not saved.
        frame_callback: Function to process each frame. Signature: frame_callback(frame, frame_number) -> processed_frame
        start_frame: First frame to process (default: 0)
        end_frame: Last frame to process (default: None, process all frames)
        skip_frames: Number of frames to skip between processing (default: 0, process every frame)
        use_pyav: Whether to use pyav for video processing (default: False, use OpenCV)
            Note: PyAV currently only supports video files, not cameras or streams
        
    Returns:
        bool: True if processing completed successfully, False otherwise
    
    Example:
        ```python
        from visionframework.utils import process_video
        
        def process_frame(frame, frame_number):
            # Example: Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Convert back to BGR for writing
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Process video file
        success = process_video(
            input_path="input.mp4",
            output_path="output.mp4",
            frame_callback=process_frame,
            start_frame=0,
            end_frame=100,  # Process only first 100 frames
            skip_frames=0  # Process every frame
        )
        
        if success:
            print("Video processing completed successfully")
        else:
            print("Video processing failed")
        ```
    """
    # Choose processor based on use_pyav parameter
    if use_pyav:
        # Check if pyav is available
        if av is None:
            print("PyAV is not installed. Falling back to OpenCV.")
            use_pyav = False
        # Check if input is a camera (PyAV doesn't support cameras)
        elif isinstance(input_path, int) or (isinstance(input_path, str) and input_path == "0"):
            print("PyAV only supports video files and streams, not cameras. Falling back to OpenCV.")
            use_pyav = False
    
    if use_pyav:
        # Use PyAVVideoProcessor for video files
        processor = PyAVVideoProcessor(input_path)
    else:
        # Use regular VideoProcessor
        processor = VideoProcessor(input_path)
    
    if not processor.open():
        return False
    
    writer = None
    if output_path:
        info = processor.get_info()
        if use_pyav:
            # Use PyAVVideoWriter
            writer = PyAVVideoWriter(output_path, fps=info["fps"], frame_size=(info["width"], info["height"]))
        else:
            # Use regular VideoWriter
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

