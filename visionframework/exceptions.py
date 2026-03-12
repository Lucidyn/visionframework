"""
Exception hierarchy for VisionFramework.
"""


class VisionFrameworkError(Exception):
    """Base exception for all framework errors."""

    def __init__(self, message: str = "", context: str = "", original_error: Exception = None):
        self.context = context
        self.original_error = original_error
        super().__init__(message)


class ConfigurationError(VisionFrameworkError):
    """Invalid or missing configuration."""


class ModelNotFoundError(VisionFrameworkError):
    """Requested model file does not exist."""


class ModelLoadError(VisionFrameworkError):
    """Failed to load or initialise a model."""


class DeviceError(VisionFrameworkError):
    """Device selection or transfer failure."""


class DetectorError(VisionFrameworkError):
    """Error during detection inference."""


class TrackerError(VisionFrameworkError):
    """Error during tracking update."""


class SegmentationError(VisionFrameworkError):
    """Error during segmentation inference."""


class PipelineError(VisionFrameworkError):
    """Error in pipeline orchestration."""


class DataFormatError(VisionFrameworkError):
    """Invalid input data format."""


class VideoProcessingError(VisionFrameworkError):
    """Error reading or writing video."""
