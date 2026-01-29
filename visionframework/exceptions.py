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


class PoseEstimationError(VisionFrameworkError):
    """Raised when pose estimation fails"""
    def __init__(self, message: str, model_type: str = None, 
                 input_shape: tuple = None, device: str = None, 
                 original_error: Exception = None):
        context = {}
        if model_type:
            context["model_type"] = model_type
        if input_shape:
            context["input_shape"] = input_shape
        if device:
            context["device"] = device
        
        super().__init__(message, context, original_error)


class CLIPProcessingError(VisionFrameworkError):
    """Raised when CLIP extraction or processing fails"""
    def __init__(self, message: str, operation: str = None, 
                 model_name: str = None, input_type: str = None, 
                 original_error: Exception = None):
        context = {}
        if operation:
            context["operation"] = operation
        if model_name:
            context["model_name"] = model_name
        if input_type:
            context["input_type"] = input_type
        
        super().__init__(message, context, original_error)


class ReIDError(VisionFrameworkError):
    """Raised when ReID (re-identification) fails"""
    def __init__(self, message: str, operation: str = None, 
                 model_path: str = None, embedding_dim: int = None, 
                 original_error: Exception = None):
        context = {}
        if operation:
            context["operation"] = operation
        if model_path:
            context["model_path"] = model_path
        if embedding_dim:
            context["embedding_dim"] = embedding_dim
        
        super().__init__(message, context, original_error)


class CacheError(VisionFrameworkError):
    """Raised when model caching operations fail"""
    def __init__(self, message: str, cache_key: str = None, 
                 operation: str = None, model_type: str = None, 
                 original_error: Exception = None):
        context = {}
        if cache_key:
            context["cache_key"] = cache_key
        if operation:
            context["operation"] = operation
        if model_type:
            context["model_type"] = model_type
        
        super().__init__(message, context, original_error)


class PipelineIntegrationError(VisionFrameworkError):
    """Raised when integrating components in the pipeline fails"""
    def __init__(self, message: str, component_name: str = None, 
                 component_type: str = None, stage: str = None, 
                 original_error: Exception = None):
        context = {}
        if component_name:
            context["component_name"] = component_name
        if component_type:
            context["component_type"] = component_type
        if stage:
            context["stage"] = stage
        
        super().__init__(message, context, original_error)


class VideoProcessingError(VisionFrameworkError):
    """Raised when video processing fails"""
    def __init__(self, message: str, video_path: str = None, 
                 operation: str = None, frame_number: int = None, 
                 original_error: Exception = None):
        context = {}
        if video_path:
            context["video_path"] = video_path
        if operation:
            context["operation"] = operation
        if frame_number is not None:
            context["frame_number"] = frame_number
        
        super().__init__(message, context, original_error)


class VideoReaderError(VideoProcessingError):
    """Raised when video reading fails"""
    def __init__(self, message: str, video_path: str = None, 
                 backend: str = None, frame_number: int = None, 
                 original_error: Exception = None):
        context = {}
        if video_path:
            context["video_path"] = video_path
        if backend:
            context["backend"] = backend
        if frame_number is not None:
            context["frame_number"] = frame_number
        
        super().__init__(message, context, original_error)


class VideoWriterError(VideoProcessingError):
    """Raised when video writing fails"""
    def __init__(self, message: str, output_path: str = None, 
                 codec: str = None, frame_size: tuple = None, 
                 original_error: Exception = None):
        context = {}
        if output_path:
            context["output_path"] = output_path
        if codec:
            context["codec"] = codec
        if frame_size:
            context["frame_size"] = frame_size
        
        super().__init__(message, context, original_error)


class SegmenterInitializationError(VisionFrameworkError):
    """Raised when segmenter initialization fails"""
    def __init__(self, message: str, segmenter_type: str = None, 
                 model_path: str = None, device: str = None, 
                 original_error: Exception = None):
        context = {}
        if segmenter_type:
            context["segmenter_type"] = segmenter_type
        if model_path:
            context["model_path"] = model_path
        if device:
            context["device"] = device
        
        super().__init__(message, context, original_error)


class SegmentationError(VisionFrameworkError):
    """Raised when segmentation fails"""
    def __init__(self, message: str, segmenter_type: str = None, 
                 input_shape: tuple = None, device: str = None, 
                 original_error: Exception = None):
        context = {}
        if segmenter_type:
            context["segmenter_type"] = segmenter_type
        if input_shape:
            context["input_shape"] = input_shape
        if device:
            context["device"] = device
        
        super().__init__(message, context, original_error)


class ConcurrentProcessingError(VisionFrameworkError):
    """Raised when concurrent processing fails"""
    def __init__(self, message: str, num_workers: int = None, 
                 batch_size: int = None, operation: str = None, 
                 original_error: Exception = None):
        context = {}
        if num_workers:
            context["num_workers"] = num_workers
        if batch_size:
            context["batch_size"] = batch_size
        if operation:
            context["operation"] = operation
        
        super().__init__(message, context, original_error)


class MemoryAllocationError(VisionFrameworkError):
    """Raised when memory allocation fails"""
    def __init__(self, message: str, requested_size: int = None, 
                 available_memory: int = None, operation: str = None, 
                 original_error: Exception = None):
        context = {}
        if requested_size:
            context["requested_size"] = requested_size
        if available_memory:
            context["available_memory"] = available_memory
        if operation:
            context["operation"] = operation
        
        super().__init__(message, context, original_error)


class TimeoutError(VisionFrameworkError):
    """Raised when an operation times out"""
    def __init__(self, message: str, operation: str = None, 
                 timeout_seconds: float = None, original_error: Exception = None):
        context = {}
        if operation:
            context["operation"] = operation
        if timeout_seconds is not None:
            context["timeout_seconds"] = timeout_seconds
        
        super().__init__(message, context, original_error)


class ROIProcessingError(VisionFrameworkError):
    """Raised when ROI (Region of Interest) processing fails"""
    def __init__(self, message: str, roi_type: str = None, 
                 roi_coordinates: list = None, operation: str = None, 
                 original_error: Exception = None):
        context = {}
        if roi_type:
            context["roi_type"] = roi_type
        if roi_coordinates:
            context["roi_coordinates"] = roi_coordinates
        if operation:
            context["operation"] = operation
        
        super().__init__(message, context, original_error)
