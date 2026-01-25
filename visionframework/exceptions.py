"""
Vision Framework Exception Classes

Provides a hierarchy of custom exceptions for better error handling and diagnostics.
Each exception class supports detailed error messages with context information.
"""


class VisionFrameworkError(Exception):
    """Base exception for all Vision Framework errors
    
    Args:
        message: Error message describing what went wrong
        context: Additional context information about the error
        original_error: Original exception that caused this error (if any)
    """
    def __init__(self, message: str, context: dict = None, original_error: Exception = None):
        self.message = message
        self.context = context or {}
        self.original_error = original_error
        
        full_message = self._format_message()
        super().__init__(full_message)
    
    def _format_message(self) -> str:
        """Format the complete error message with context"""
        full_message = f"{self.__class__.__name__}: {self.message}"
        
        if self.context:
            context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
            full_message += f" [{context_str}]"
        
        if self.original_error:
            full_message += f" (Original error: {str(self.original_error)})"
        
        return full_message


class DetectorInitializationError(VisionFrameworkError):
    """Raised when detector initialization fails"""
    def __init__(self, message: str, model_type: str = None, model_path: str = None, 
                 device: str = None, original_error: Exception = None):
        context = {}
        if model_type:
            context["model_type"] = model_type
        if model_path:
            context["model_path"] = model_path
        if device:
            context["device"] = device
        
        super().__init__(message, context, original_error)


class DetectorInferenceError(VisionFrameworkError):
    """Raised when detector inference fails"""
    def __init__(self, message: str, model_type: str = None, device: str = None, 
                 input_shape: tuple = None, original_error: Exception = None):
        context = {}
        if model_type:
            context["model_type"] = model_type
        if device:
            context["device"] = device
        if input_shape:
            context["input_shape"] = input_shape
        
        super().__init__(message, context, original_error)


class TrackerInitializationError(VisionFrameworkError):
    """Raised when tracker initialization fails"""
    def __init__(self, message: str, tracker_type: str = None, 
                 original_error: Exception = None):
        context = {}
        if tracker_type:
            context["tracker_type"] = tracker_type
        
        super().__init__(message, context, original_error)


class TrackerUpdateError(VisionFrameworkError):
    """Raised when tracker update fails"""
    def __init__(self, message: str, tracker_type: str = None, frame_idx: int = None, 
                 num_detections: int = None, original_error: Exception = None):
        context = {}
        if tracker_type:
            context["tracker_type"] = tracker_type
        if frame_idx is not None:
            context["frame_idx"] = frame_idx
        if num_detections is not None:
            context["num_detections"] = num_detections
        
        super().__init__(message, context, original_error)


class ConfigurationError(VisionFrameworkError):
    """Raised when configuration is invalid"""
    def __init__(self, message: str, config_key: str = None, config_value: any = None, 
                 expected_type: type = None, original_error: Exception = None):
        context = {}
        if config_key:
            context["config_key"] = config_key
        if config_value is not None:
            context["config_value"] = config_value
        if expected_type:
            context["expected_type"] = expected_type.__name__
        
        super().__init__(message, context, original_error)


class ModelNotFoundError(VisionFrameworkError):
    """Raised when a model file is not found"""
    def __init__(self, message: str, model_path: str = None, 
                 original_error: Exception = None):
        context = {}
        if model_path:
            context["model_path"] = model_path
        
        super().__init__(message, context, original_error)


class ModelLoadError(VisionFrameworkError):
    """Raised when model loading fails"""
    def __init__(self, message: str, model_path: str = None, device: str = None, 
                 original_error: Exception = None):
        context = {}
        if model_path:
            context["model_path"] = model_path
        if device:
            context["device"] = device
        
        super().__init__(message, context, original_error)


class DeviceError(VisionFrameworkError):
    """Raised when device (CPU/GPU) is unavailable or invalid"""
    def __init__(self, message: str, requested_device: str = None, 
                 available_devices: list = None, original_error: Exception = None):
        context = {}
        if requested_device:
            context["requested_device"] = requested_device
        if available_devices:
            context["available_devices"] = available_devices
        
        super().__init__(message, context, original_error)


class DependencyError(VisionFrameworkError):
    """Raised when a required dependency is missing"""
    def __init__(self, message: str, dependency_name: str = None, 
                 expected_version: str = None, installation_command: str = None, 
                 original_error: Exception = None):
        context = {}
        if dependency_name:
            context["dependency_name"] = dependency_name
        if expected_version:
            context["expected_version"] = expected_version
        if installation_command:
            context["installation_command"] = installation_command
        
        super().__init__(message, context, original_error)


class DataFormatError(VisionFrameworkError):
    """Raised when input data format is invalid"""
    def __init__(self, message: str, expected_format: str = None, 
                 actual_format: str = None, input_shape: tuple = None, 
                 original_error: Exception = None):
        context = {}
        if expected_format:
            context["expected_format"] = expected_format
        if actual_format:
            context["actual_format"] = actual_format
        if input_shape:
            context["input_shape"] = input_shape
        
        super().__init__(message, context, original_error)


class ProcessingError(VisionFrameworkError):
    """Raised when processing (detection, tracking, etc.) fails"""
    def __init__(self, message: str, processing_stage: str = None, 
                 input_shape: tuple = None, original_error: Exception = None):
        context = {}
        if processing_stage:
            context["processing_stage"] = processing_stage
        if input_shape:
            context["input_shape"] = input_shape
        
        super().__init__(message, context, original_error)


class AnnotationError(VisionFrameworkError):
    """Raised when annotation generation or processing fails"""
    def __init__(self, message: str, annotation_type: str = None, 
                 output_format: str = None, original_error: Exception = None):
        context = {}
        if annotation_type:
            context["annotation_type"] = annotation_type
        if output_format:
            context["output_format"] = output_format
        
        super().__init__(message, context, original_error)


class PerformanceError(VisionFrameworkError):
    """Raised when performance-related issues occur"""
    def __init__(self, message: str, metric: str = None, 
                 value: float = None, threshold: float = None, 
                 original_error: Exception = None):
        context = {}
        if metric:
            context["metric"] = metric
        if value is not None:
            context["value"] = value
        if threshold is not None:
            context["threshold"] = threshold
        
        super().__init__(message, context, original_error)


class VisualizationError(VisionFrameworkError):
    """Raised when visualization fails"""
    def __init__(self, message: str, visualization_type: str = None, 
                 output_path: str = None, original_error: Exception = None):
        context = {}
        if visualization_type:
            context["visualization_type"] = visualization_type
        if output_path:
            context["output_path"] = output_path
        
        super().__init__(message, context, original_error)


class BatchProcessingError(VisionFrameworkError):
    """Raised when batch processing fails"""
    def __init__(self, message: str, batch_size: int = None, 
                 num_processed: int = None, num_failed: int = None, 
                 original_error: Exception = None):
        context = {}
        if batch_size:
            context["batch_size"] = batch_size
        if num_processed is not None:
            context["num_processed"] = num_processed
        if num_failed is not None:
            context["num_failed"] = num_failed
        
        super().__init__(message, context, original_error)


class ExportError(VisionFrameworkError):
    """Raised when exporting results fails"""
    def __init__(self, message: str, output_format: str = None, 
                 output_path: str = None, original_error: Exception = None):
        context = {}
        if output_format:
            context["output_format"] = output_format
        if output_path:
            context["output_path"] = output_path
        
        super().__init__(message, context, original_error)
