"""
Base class for all vision modules

This module provides the base class for all vision processing modules,
including configuration validation, error handling utilities, and common methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Callable
import numpy as np
from functools import wraps


class BaseModule(ABC):
    """
    Base class for all vision processing modules
    
    This class provides:
    - Configuration management and validation
    - Common initialization and reset functionality
    - Unified error handling utilities
    - Configuration update methods
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base module
        
        Args:
            config: Configuration dictionary. If None, an empty dictionary is used.
                    Subclasses should override validate_config() to validate specific parameters.
        
        Note:
            Configuration validation is performed automatically if validate_config() is implemented.
        """
        self.config = config or {}
        self.is_initialized = False
        
        # Validate configuration if validation method exists
        if hasattr(self, 'validate_config'):
            is_valid, error_msg = self.validate_config(self.config)
            if not is_valid:
                from ..utils.logger import get_logger
                logger = get_logger(__name__)
                logger.warning(f"Invalid configuration: {error_msg}")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the module
        
        This method must be implemented by subclasses to perform any necessary
        initialization, such as loading models, setting up resources, etc.
        
        Returns:
            bool: True if initialization successful, False otherwise.
                  On failure, error should be logged using the logger.
        
        Raises:
            RuntimeError: If initialization fails due to runtime issues.
            ValueError: If configuration is invalid.
            ImportError: If required dependencies are missing.
        """
        pass
    
    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """
        Process input data
        
        This method must be implemented by subclasses to process input data
        and return results.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        
        Returns:
            Processed results. The exact type depends on the subclass implementation.
        
        Note:
            Subclasses should check is_initialized before processing and
            return appropriate default values (e.g., empty list) if not initialized.
        """
        pass
    
    def reset(self) -> None:
        """
        Reset the module state
        
        This method resets the module to an uninitialized state, allowing
        for re-initialization with new configuration if needed.
        
        Subclasses should override this method to reset any module-specific state.
        """
        self.is_initialized = False
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration
        
        Returns:
            Dict[str, Any]: A copy of the current configuration dictionary.
                          Modifying the returned dictionary will not affect the module's configuration.
        """
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration
        
        This method updates the module's configuration with new values.
        If the module is already initialized, it will be reset and re-initialized
        with the new configuration.
        
        Args:
            new_config: Dictionary containing new configuration values.
                       These values will be merged with the existing configuration.
        
        Note:
            Configuration validation is performed automatically if validate_config() is implemented.
        """
        self.config.update(new_config)
        if self.is_initialized:
            self.reset()
            self.initialize()
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate configuration parameters
        
        This method can be overridden by subclasses to implement custom
        configuration validation logic.
        
        Args:
            config: Configuration dictionary to validate
        
        Returns:
            Tuple[bool, Optional[str]]: 
                - (True, None) if configuration is valid
                - (False, error_message) if configuration is invalid
                  error_message should describe what is wrong with the configuration
        
        Example:
            ```python
            def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
                if "conf_threshold" in config:
                    threshold = config["conf_threshold"]
                    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
                        return False, "conf_threshold must be a number between 0 and 1"
                return True, None
            ```
        """
        return True, None
    
    @staticmethod
    def handle_errors(default_return: Any = None, log_errors: bool = True):
        """
        Decorator for unified error handling
        
        This decorator provides a unified way to handle errors in module methods.
        It catches exceptions, logs them, and returns a default value.
        
        Args:
            default_return: Default value to return on error. If None, the function
                           should specify its own default return type.
            log_errors: Whether to log errors. Default is True.
        
        Returns:
            Decorated function that handles errors uniformly
        
        Example:
            ```python
            @BaseModule.handle_errors(default_return=[])
            def detect(self, image: np.ndarray) -> List[Detection]:
                # Implementation
                return detections
            ```
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                try:
                    return func(self, *args, **kwargs)
                except Exception as e:
                    if log_errors:
                        from ..utils.logger import get_logger
                        logger = get_logger(self.__class__.__module__)
                        logger.error(
                            f"Error in {self.__class__.__name__}.{func.__name__}: {e}",
                            exc_info=True
                        )
                    return default_return
            return wrapper
        return decorator

