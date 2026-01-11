"""
Performance analysis and monitoring utilities
"""

import time
from typing import Dict, List, Optional, Any
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    fps: float = 0.0
    avg_fps: float = 0.0
    min_fps: float = float('inf')
    max_fps: float = 0.0
    frame_count: int = 0
    total_time: float = 0.0
    avg_time_per_frame: float = 0.0
    min_time_per_frame: float = float('inf')
    max_time_per_frame: float = 0.0
    detection_time: float = 0.0
    tracking_time: float = 0.0
    visualization_time: float = 0.0


class PerformanceMonitor:
    """Monitor and analyze performance metrics"""
    
    def __init__(self, window_size: int = 30):
        """
        Initialize performance monitor
        
        Args:
            window_size: Size of the sliding window for FPS calculation
        """
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self.detection_times = deque(maxlen=window_size)
        self.tracking_times = deque(maxlen=window_size)
        self.visualization_times = deque(maxlen=window_size)
        
        self.start_time = None
        self.last_frame_time = None
        self.frame_count = 0
        self.total_detection_time = 0.0
        self.total_tracking_time = 0.0
        self.total_visualization_time = 0.0
        
        self.current_fps = 0.0
        self.avg_fps = 0.0
    
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        self.last_frame_time = self.start_time
    
    def tick(self):
        """Mark a frame processing tick"""
        current_time = time.time()
        
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
            self.current_fps = 1.0 / frame_time if frame_time > 0 else 0.0
        
        self.last_frame_time = current_time
        self.frame_count += 1
        
        if self.frame_count > 0:
            total_elapsed = current_time - self.start_time
            self.avg_fps = self.frame_count / total_elapsed if total_elapsed > 0 else 0.0
    
    def record_detection_time(self, detection_time: float):
        """Record detection processing time"""
        self.detection_times.append(detection_time)
        self.total_detection_time += detection_time
    
    def record_tracking_time(self, tracking_time: float):
        """Record tracking processing time"""
        self.tracking_times.append(tracking_time)
        self.total_tracking_time += tracking_time
    
    def record_visualization_time(self, visualization_time: float):
        """Record visualization processing time"""
        self.visualization_times.append(visualization_time)
        self.total_visualization_time += visualization_time
    
    def get_current_fps(self) -> float:
        """Get current FPS"""
        return self.current_fps
    
    def get_average_fps(self) -> float:
        """Get average FPS"""
        return self.avg_fps
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics"""
        if len(self.frame_times) == 0:
            return PerformanceMetrics()
        
        frame_times_list = list(self.frame_times)
        fps_list = [1.0 / t for t in frame_times_list if t > 0]
        
        metrics = PerformanceMetrics(
            fps=self.current_fps,
            avg_fps=self.avg_fps,
            min_fps=min(fps_list) if fps_list else 0.0,
            max_fps=max(fps_list) if fps_list else 0.0,
            frame_count=self.frame_count,
            total_time=time.time() - self.start_time if self.start_time else 0.0,
            avg_time_per_frame=sum(frame_times_list) / len(frame_times_list),
            min_time_per_frame=min(frame_times_list),
            max_time_per_frame=max(frame_times_list),
            detection_time=sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0.0,
            tracking_time=sum(self.tracking_times) / len(self.tracking_times) if self.tracking_times else 0.0,
            visualization_time=sum(self.visualization_times) / len(self.visualization_times) if self.visualization_times else 0.0
        )
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary as dictionary"""
        metrics = self.get_metrics()
        return {
            "fps": {
                "current": metrics.fps,
                "average": metrics.avg_fps,
                "min": metrics.min_fps,
                "max": metrics.max_fps
            },
            "frame_count": metrics.frame_count,
            "total_time": metrics.total_time,
            "time_per_frame": {
                "average": metrics.avg_time_per_frame,
                "min": metrics.min_time_per_frame,
                "max": metrics.max_time_per_frame
            },
            "component_times": {
                "detection": metrics.detection_time,
                "tracking": metrics.tracking_time,
                "visualization": metrics.visualization_time
            }
        }
    
    def print_summary(self):
        """Print performance summary"""
        summary = self.get_summary()
        print("\n" + "=" * 50)
        print("Performance Summary")
        print("=" * 50)
        print(f"FPS: {summary['fps']['current']:.2f} (avg: {summary['fps']['average']:.2f}, "
              f"min: {summary['fps']['min']:.2f}, max: {summary['fps']['max']:.2f})")
        print(f"Frames processed: {summary['frame_count']}")
        print(f"Total time: {summary['total_time']:.2f}s")
        print(f"Time per frame: {summary['time_per_frame']['average']*1000:.2f}ms "
              f"(min: {summary['time_per_frame']['min']*1000:.2f}ms, "
              f"max: {summary['time_per_frame']['max']*1000:.2f}ms)")
        print(f"Component times:")
        print(f"  - Detection: {summary['component_times']['detection']*1000:.2f}ms")
        print(f"  - Tracking: {summary['component_times']['tracking']*1000:.2f}ms")
        print(f"  - Visualization: {summary['component_times']['visualization']*1000:.2f}ms")
        print("=" * 50)
    
    def reset(self):
        """Reset all metrics"""
        self.frame_times.clear()
        self.detection_times.clear()
        self.tracking_times.clear()
        self.visualization_times.clear()
        self.start_time = None
        self.last_frame_time = None
        self.frame_count = 0
        self.total_detection_time = 0.0
        self.total_tracking_time = 0.0
        self.total_visualization_time = 0.0
        self.current_fps = 0.0
        self.avg_fps = 0.0


class Timer:
    """Simple context manager for timing code blocks"""
    
    def __init__(self, name: str = "Operation"):
        """
        Initialize timer
        
        Args:
            name: Name of the operation being timed
        """
        self.name = name
        self.start_time = None
        self.elapsed_time = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_time = time.time() - self.start_time
        return False
    
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds"""
        return self.elapsed_time

