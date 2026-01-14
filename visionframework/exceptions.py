"""
Vision Framework Exception Classes

Provides a hierarchy of custom exceptions for better error handling and diagnostics.
"""


class VisionFrameworkError(Exception):
    """Base exception for all Vision Framework errors"""
    pass


class DetectorInitializationError(VisionFrameworkError):
    """Raised when detector initialization fails"""
    pass


class DetectorInferenceError(VisionFrameworkError):
    """Raised when detector inference fails"""
    pass


class TrackerInitializationError(VisionFrameworkError):
    """Raised when tracker initialization fails"""
    pass


class TrackerUpdateError(VisionFrameworkError):
    """Raised when tracker update fails"""
    pass


class ConfigurationError(VisionFrameworkError):
    """Raised when configuration is invalid"""
    pass


class ModelNotFoundError(VisionFrameworkError):
    """Raised when a model file is not found"""
    pass


class ModelLoadError(VisionFrameworkError):
    """Raised when model loading fails"""
    pass


class DeviceError(VisionFrameworkError):
    """Raised when device (CPU/GPU) is unavailable or invalid"""
    pass


class DependencyError(VisionFrameworkError):
    """Raised when a required dependency is missing"""
    pass


class DataFormatError(VisionFrameworkError):
    """Raised when input data format is invalid"""
    pass


class ProcessingError(VisionFrameworkError):
    """Raised when processing (detection, tracking, etc.) fails"""
    pass
