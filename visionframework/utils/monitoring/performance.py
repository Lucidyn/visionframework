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
    """
    Performance metrics container with extended metrics
    """
    # FPS metrics
    fps: float = 0.0
    avg_fps: float = 0.0
    min_fps: float = 0.0
    max_fps: float = 0.0
    
    # Time metrics
    frame_count: int = 0
    total_time: float = 0.0
    avg_time_per_frame: float = 0.0
    min_time_per_frame: float = 0.0
    max_time_per_frame: float = 0.0
    
    # Component times (in milliseconds)
    component_times: Dict[str, float] = field(default_factory=dict)
    
    # Resource metrics
    avg_memory_usage: float = 0.0  # in MB
    peak_memory_usage: float = 0.0  # in MB
    
    # Disk I/O metrics
    avg_disk_reads: float = 0.0  # in bytes
    avg_disk_writes: float = 0.0  # in bytes
    
    # Network I/O metrics
    avg_network_sent: float = 0.0  # in bytes
    avg_network_recv: float = 0.0  # in bytes
    
    # GPU metrics (if available)
    gpu_memory_usage: float = 0.0  # in MB
    gpu_utilization: float = 0.0  # in percent


class PerformanceMonitor:
    """
    Enhanced performance monitor with extended metrics and resource monitoring
    """
    
    def __init__(self, window_size: int = 30, enabled_metrics: Optional[List[str]] = None, enable_gpu_monitoring: bool = False, enable_disk_monitoring: bool = False, enable_network_monitoring: bool = False):
        """
        Initialize performance monitor
        
        Args:
            window_size: Size of the sliding window for metrics calculation
            enabled_metrics: List of metrics to enable (None for all)
            enable_gpu_monitoring: Whether to enable GPU monitoring (default: False)
            enable_disk_monitoring: Whether to enable disk I/O monitoring (default: False)
            enable_network_monitoring: Whether to enable network I/O monitoring (default: False)
        """
        self.window_size = window_size
        self.enabled_metrics = enabled_metrics
        self.frame_times = deque(maxlen=window_size)
        self.component_times = {
            'detection': deque(maxlen=window_size),
            'tracking': deque(maxlen=window_size),
            'visualization': deque(maxlen=window_size),
            'pose_estimation': deque(maxlen=window_size),
            'clip_extraction': deque(maxlen=window_size),
            'reid': deque(maxlen=window_size),
            'batch_processing': deque(maxlen=window_size),
            'memory_management': deque(maxlen=window_size),
            'io_operations': deque(maxlen=window_size),
            'network_operations': deque(maxlen=window_size),
        }
        
        # Memory tracking
        self.memory_usages = deque(maxlen=window_size)
        self.peak_memory_usage = 0.0
        self.memory_allocations = deque(maxlen=window_size)
        self.memory_deallocations = deque(maxlen=window_size)
        
        # Disk I/O tracking
        self.disk_reads = deque(maxlen=window_size)
        self.disk_writes = deque(maxlen=window_size)
        self.enable_disk_monitoring = enable_disk_monitoring
        
        # Network I/O tracking
        self.network_bytes_sent = deque(maxlen=window_size)
        self.network_bytes_recv = deque(maxlen=window_size)
        self.enable_network_monitoring = enable_network_monitoring
        
        # GPU tracking (disabled by default)
        self.gpu_memory_usages = deque(maxlen=window_size)
        self.gpu_utilizations = deque(maxlen=window_size)
        self.gpu_available = False
        self._nvml = False
        self.enable_gpu_monitoring = enable_gpu_monitoring
        
        # General timing
        self.start_time = None
        self.last_frame_time = None
        self.frame_count = 0
        
        # Initialize system monitors
        self._init_system_monitors()
    
    def start(self):
        """Start timing and resource monitoring"""
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        self.frame_count = 0
        self.peak_memory_usage = 0.0
        
        # Initialize GPU monitoring only if enabled and starting
        if self.enable_gpu_monitoring:
            self._init_gpu_monitoring()
    
    def _init_system_monitors(self):
        """
        Initialize system monitors
        """
        # Initialize GPU monitoring
        if self.enable_gpu_monitoring:
            self._init_gpu_monitoring()
        
        # Initialize disk monitoring
        if self.enable_disk_monitoring:
            self._init_disk_monitoring()
        
        # Initialize network monitoring
        if self.enable_network_monitoring:
            self._init_network_monitoring()
    
    def _init_gpu_monitoring(self):
        """
        Initialize GPU monitoring if available
        """
        self.gpu_available = False
        self._nvml = False
        try:
            import torch
            if torch.cuda.is_available():
                self.gpu_available = True
                # Only import nvidia-ml-py3 if needed
                try:
                    from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
                    nvmlInit()
                    self._nvml = True
                    self._gpu_handle = nvmlDeviceGetHandleByIndex(0)  # Assume first GPU
                except ImportError:
                    self._nvml = False
                except Exception as e:
                    # Handle any NVML initialization errors
                    self._nvml = False
        except ImportError:
            # torch is not available, GPU monitoring is disabled
            self.gpu_available = False
        except Exception as e:
            # Any other error during GPU monitoring initialization
            self.gpu_available = False
    
    def _init_disk_monitoring(self):
        """
        Initialize disk monitoring
        """
        try:
            import psutil
            self._psutil_available = True
            self._disk_io = psutil.disk_io_counters()
        except ImportError:
            self._psutil_available = False
    
    def _init_network_monitoring(self):
        """
        Initialize network monitoring
        """
        try:
            import psutil
            self._psutil_available = True
            self._network_io = psutil.net_io_counters()
        except ImportError:
            self._psutil_available = False
    
    def _record_gpu_usage(self):
        """Record GPU memory and utilization"""
        if self.gpu_available and self._nvml:
            try:
                from pynvml import nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
                # Get GPU memory
                memory_info = nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_memory_mb = memory_info.used / 1024 / 1024  # Convert to MB
                self.gpu_memory_usages.append(gpu_memory_mb)
                
                # Get GPU utilization
                util_info = nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_util = util_info.gpu
                self.gpu_utilizations.append(gpu_util)
            except Exception as e:
                # Silently handle any GPU monitoring errors
                pass
    
    def start(self):
        """Start timing and resource monitoring"""
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        self.frame_count = 0
        self.peak_memory_usage = 0.0
    
    def tick(self):
        """
        Mark a frame processing tick and record resource usage
        """
        current_time = time.time()
        
        if self.last_frame_time is not None:
            frame_time = current_time - self.last_frame_time
            self.frame_times.append(frame_time)
        
        self.frame_count += 1
        self.last_frame_time = current_time
        
        # Record memory usage
        self._record_memory_usage()
        
        # Record disk usage if enabled
        if self.enable_disk_monitoring:
            self._record_disk_usage()
        
        # Record network usage if enabled
        if self.enable_network_monitoring:
            self._record_network_usage()
        
        # Record GPU usage if available
        if self.gpu_available:
            self._record_gpu_usage()
    
    def _record_memory_usage(self):
        """
        Record current memory usage
        """
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024  # Convert to MB
            self.memory_usages.append(memory_mb)
            # Update peak memory
            if memory_mb > self.peak_memory_usage:
                self.peak_memory_usage = memory_mb
        except ImportError:
            pass
    
    def _record_disk_usage(self):
        """
        Record current disk I/O usage
        """
        if not self.enable_disk_monitoring or not hasattr(self, '_psutil_available') or not self._psutil_available:
            return
        
        try:
            import psutil
            disk_io = psutil.disk_io_counters()
            if hasattr(self, '_disk_io'):
                read_bytes = disk_io.read_bytes - self._disk_io.read_bytes
                write_bytes = disk_io.write_bytes - self._disk_io.write_bytes
                self.disk_reads.append(read_bytes)
                self.disk_writes.append(write_bytes)
            self._disk_io = disk_io
        except Exception:
            pass
    
    def _record_network_usage(self):
        """
        Record current network I/O usage
        """
        if not self.enable_network_monitoring or not hasattr(self, '_psutil_available') or not self._psutil_available:
            return
        
        try:
            import psutil
            network_io = psutil.net_io_counters()
            if hasattr(self, '_network_io'):
                bytes_sent = network_io.bytes_sent - self._network_io.bytes_sent
                bytes_recv = network_io.bytes_recv - self._network_io.bytes_recv
                self.network_bytes_sent.append(bytes_sent)
                self.network_bytes_recv.append(bytes_recv)
            self._network_io = network_io
        except Exception:
            pass
    
    def _record_gpu_usage(self):
        """Record GPU memory and utilization"""
        if self.gpu_available and hasattr(self, '_nvml') and self._nvml:
            try:
                from pynvml import nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
                # Get GPU memory
                memory_info = nvmlDeviceGetMemoryInfo(self._gpu_handle)
                gpu_memory_mb = memory_info.used / 1024 / 1024  # Convert to MB
                self.gpu_memory_usages.append(gpu_memory_mb)
                
                # Get GPU utilization
                util_info = nvmlDeviceGetUtilizationRates(self._gpu_handle)
                gpu_util = util_info.gpu
                self.gpu_utilizations.append(gpu_util)
            except Exception:
                pass
    
    def record_component_time(self, component: str, elapsed: float):
        """Record component processing time"""
        # Only record time for known components
        if component in self.component_times:
            self.component_times[component].append(elapsed)
    
    def get_metrics(self) -> PerformanceMetrics:
        """
        Get comprehensive performance metrics
        """
        if len(self.frame_times) == 0:
            return PerformanceMetrics()
        
        frame_times_list = list(self.frame_times)
        total_elapsed = time.time() - self.start_time if self.start_time else 0.0
        
        # Calculate FPS metrics
        fps_values = [1.0 / ft if ft > 0 else 0.0 for ft in frame_times_list]
        current_fps = fps_values[-1] if fps_values else 0.0
        avg_fps = len(self.frame_times) / total_elapsed if total_elapsed > 0 else 0.0
        min_fps = min(fps_values) if fps_values else 0.0
        max_fps = max(fps_values) if fps_values else 0.0
        
        # Calculate time metrics
        avg_time_per_frame = sum(frame_times_list) / len(frame_times_list)
        min_time_per_frame = min(frame_times_list)
        max_time_per_frame = max(frame_times_list)
        
        # Calculate component average times
        component_avg_times = {}
        for component, times in self.component_times.items():
            if times:
                component_avg_times[component] = sum(times) / len(times) * 1000  # Convert to milliseconds
        
        # Calculate memory metrics
        avg_memory_usage = 0.0
        if self.memory_usages:
            avg_memory_usage = sum(self.memory_usages) / len(self.memory_usages)
        
        # Calculate disk metrics
        avg_disk_reads = 0.0
        avg_disk_writes = 0.0
        if self.disk_reads:
            avg_disk_reads = sum(self.disk_reads) / len(self.disk_reads)
        if self.disk_writes:
            avg_disk_writes = sum(self.disk_writes) / len(self.disk_writes)
        
        # Calculate network metrics
        avg_network_sent = 0.0
        avg_network_recv = 0.0
        if self.network_bytes_sent:
            avg_network_sent = sum(self.network_bytes_sent) / len(self.network_bytes_sent)
        if self.network_bytes_recv:
            avg_network_recv = sum(self.network_bytes_recv) / len(self.network_bytes_recv)
        
        # Calculate GPU metrics
        gpu_memory_usage = 0.0
        gpu_utilization = 0.0
        if self.gpu_memory_usages:
            gpu_memory_usage = sum(self.gpu_memory_usages) / len(self.gpu_memory_usages)
        if self.gpu_utilizations:
            gpu_utilization = sum(self.gpu_utilizations) / len(self.gpu_utilizations)
        
        metrics = PerformanceMetrics(
            # FPS metrics
            fps=current_fps,
            avg_fps=avg_fps,
            min_fps=min_fps,
            max_fps=max_fps,
            
            # Time metrics
            frame_count=self.frame_count,
            total_time=total_elapsed,
            avg_time_per_frame=avg_time_per_frame,
            min_time_per_frame=min_time_per_frame,
            max_time_per_frame=max_time_per_frame,
            
            # Component times
            component_times=component_avg_times,
            
            # Memory metrics
            avg_memory_usage=avg_memory_usage,
            peak_memory_usage=self.peak_memory_usage,
            
            # Disk metrics
            avg_disk_reads=avg_disk_reads,
            avg_disk_writes=avg_disk_writes,
            
            # Network metrics
            avg_network_sent=avg_network_sent,
            avg_network_recv=avg_network_recv,
            
            # GPU metrics
            gpu_memory_usage=gpu_memory_usage,
            gpu_utilization=gpu_utilization
        )
        
        return metrics
    
    def get_detailed_report(self) -> Dict[str, Any]:
        """
        Get detailed performance report as dictionary
        """
        metrics = self.get_metrics()
        report = {
            'timestamp': time.time(),
            'fps': {
                'current': metrics.fps,
                'average': metrics.avg_fps,
                'min': metrics.min_fps,
                'max': metrics.max_fps
            },
            'frame_times': {
                'average': metrics.avg_time_per_frame,
                'min': metrics.min_time_per_frame,
                'max': metrics.max_time_per_frame
            },
            'component_times_ms': metrics.component_times,
            'memory': {
                'average_mb': metrics.avg_memory_usage,
                'peak_mb': metrics.peak_memory_usage
            },
            'disk_io': {
                'avg_reads_bytes': metrics.avg_disk_reads,
                'avg_writes_bytes': metrics.avg_disk_writes
            },
            'network_io': {
                'avg_sent_bytes': metrics.avg_network_sent,
                'avg_recv_bytes': metrics.avg_network_recv
            },
            'general': {
                'frame_count': metrics.frame_count,
                'total_time': metrics.total_time
            }
        }
        
        # Add GPU metrics if available
        if self.gpu_available:
            report['gpu'] = {
                'memory_usage_mb': metrics.gpu_memory_usage,
                'utilization_percent': metrics.gpu_utilization
            }
        
        return report
    
    def reset(self):
        """
        Reset all metrics and resource tracking
        """
        self.frame_times.clear()
        for deque_ in self.component_times.values():
            deque_.clear()
        self.memory_usages.clear()
        self.memory_allocations.clear()
        self.memory_deallocations.clear()
        self.disk_reads.clear()
        self.disk_writes.clear()
        self.network_bytes_sent.clear()
        self.network_bytes_recv.clear()
        self.gpu_memory_usages.clear()
        self.gpu_utilizations.clear()
        self.peak_memory_usage = 0.0
        self.start_time = None
        self.last_frame_time = None
        self.frame_count = 0


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

