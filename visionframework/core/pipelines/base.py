"""
Base pipeline class for Vision Framework
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from ..base import BaseModule
from ...utils.monitoring.logger import get_logger
from ...utils.monitoring.performance import PerformanceMonitor
from ...utils.memory import create_memory_pool, optimize_memory_usage

logger = get_logger(__name__)


class BasePipeline(BaseModule):
    """
    Base pipeline class for all vision pipelines
    
    This class provides the basic functionality for all vision pipelines,
    including initialization, configuration, and common processing methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base pipeline
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config or {})
        
        # Set default config values
        self.config.setdefault("enable_performance_monitoring", False)
        self.config.setdefault("performance_metrics", [])
        
        # Initialize attributes
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self._memory_pools: Dict[str, Any] = {}
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate pipeline configuration
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Tuple[bool, Optional[str]]: (valid, error_message)
        """
        # Validate enable_performance_monitoring
        if "enable_performance_monitoring" in config:
            enable_perf = config["enable_performance_monitoring"]
            if not isinstance(enable_perf, bool):
                return False, f"enable_performance_monitoring must be a boolean, got {type(enable_perf).__name__}"
        
        # Validate performance_metrics
        if "performance_metrics" in config:
            perf_metrics = config["performance_metrics"]
            if not isinstance(perf_metrics, list):
                return False, f"performance_metrics must be a list, got {type(perf_metrics).__name__}"
        
        return True, None
    
    def initialize(self) -> bool:
        """
        Initialize pipeline components
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Initialize performance monitor if enabled
            if self.config.get("enable_performance_monitoring", False):
                logger.info("Initializing performance monitor...")
                self.performance_monitor = PerformanceMonitor(
                    metrics=self.config.get("performance_metrics", [])
                )
            
            # Initialize memory pools
            self._initialize_memory_pools()
            
            self.is_initialized = True
            logger.info("Base pipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize base pipeline: {e}")
            return False
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process a single image
        
        Args:
            image: Input image
        
        Returns:
            Dict[str, Any]: Processing results
        """
        raise NotImplementedError("Subclasses must implement process method")
    
    def process_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Process multiple images in batch
        
        Args:
            images: List of input images
        
        Returns:
            List[Dict[str, Any]]: List of processing results
        """
        raise NotImplementedError("Subclasses must implement process_batch method")
    
    def process_video(self, input_source: Any, output_path: Optional[str] = None) -> bool:
        """
        Process video
        
        Args:
            input_source: Video source
            output_path: Output video path
        
        Returns:
            bool: True if processing successful
        """
        raise NotImplementedError("Subclasses must implement process_video method")
    
    def reset(self) -> None:
        """
        Reset pipeline state
        """
        super().reset()
        # Reset performance monitor
        if self.performance_monitor:
            self.performance_monitor.reset()
    
    def _initialize_memory_pools(self) -> None:
        """
        Initialize memory pools for efficient memory allocation
        """
        try:
            # Create memory pools for different sizes
            self._memory_pools = {
                "small": create_memory_pool(size=1024 * 1024, count=10),  # 1MB * 10
                "medium": create_memory_pool(size=4 * 1024 * 1024, count=5),  # 4MB * 5
                "large": create_memory_pool(size=16 * 1024 * 1024, count=3)  # 16MB * 3
            }
            logger.debug("Memory pools initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize memory pools: {e}")
            self._memory_pools = {}
    
    def _acquire_memory(self, size: int) -> Optional[np.ndarray]:
        """
        Acquire memory from pool
        
        Args:
            size: Memory size in bytes
        
        Returns:
            Optional[np.ndarray]: Allocated memory
        """
        try:
            if size <= 1024 * 1024:
                pool = self._memory_pools.get("small")
            elif size <= 4 * 1024 * 1024:
                pool = self._memory_pools.get("medium")
            else:
                pool = self._memory_pools.get("large")
            
            if pool:
                return pool.acquire()
            return None
        except Exception as e:
            logger.warning(f"Failed to acquire memory: {e}")
            return None
    
    def _release_memory(self, memory: np.ndarray) -> None:
        """
        Release memory back to pool
        
        Args:
            memory: Memory to release
        """
        try:
            # Simple implementation - in a real pool system, you would return to the appropriate pool
            pass
        except Exception as e:
            logger.warning(f"Failed to release memory: {e}")
