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
    frame_count: int = 0
    total_time: float = 0.0
    avg_time_per_frame: float = 0.0


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
        self.component_times = {
            'detection': deque(maxlen=window_size),
            'tracking': deque(maxlen=window_size),
            'visualization': deque(maxlen=window_size)
        }
        
        self.start_time = None
        self.last_frame_time = None
    
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
        
        self.last_frame_time = current_time
    
    def record_component_time(self, component: str, elapsed: float):
        """Record component processing time"""
        if component in self.component_times:
            self.component_times[component].append(elapsed)
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get comprehensive performance metrics"""
        if len(self.frame_times) == 0:
            return PerformanceMetrics()
        
        frame_times_list = list(self.frame_times)
        total_elapsed = time.time() - self.start_time if self.start_time else 0.0
        
        # Calculate FPS
        current_fps = 1.0 / frame_times_list[-1] if frame_times_list[-1] > 0 else 0.0
        avg_fps = len(self.frame_times) / total_elapsed if total_elapsed > 0 else 0.0
        
        metrics = PerformanceMetrics(
            fps=current_fps,
            avg_fps=avg_fps,
            frame_count=len(self.frame_times),
            total_time=total_elapsed,
            avg_time_per_frame=sum(frame_times_list) / len(frame_times_list)
        )
        
        return metrics
    
    def reset(self):
        """Reset all metrics"""
        self.frame_times.clear()
        for deque_ in self.component_times.values():
            deque_.clear()
        self.start_time = None
        self.last_frame_time = None


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

